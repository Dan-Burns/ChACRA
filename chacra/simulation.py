# -*- coding: utf-8 -*-
"""
Simulation setup utilities for ChACRA.

Provides helpers for preparing OpenMM systems and running energy minimization.
All OpenMM imports are deferred to each method so that users who only need
the analysis side of ChACRA are not required to have OpenMM installed.
"""

import os


def fix_pdb(input_pdb, output_pdb, pH=7.0, keep_water=False, replace_nonstandard_resis=True):
    """
    PDBFixer convenience function.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_pdb : str
        Path to write the fixed PDB file.
    pH : float
        pH used to determine protonation states when adding missing hydrogens.
    keep_water : bool
        If True, retain crystallographic water molecules.
    replace_nonstandard_resis : bool
        If True, replace nonstandard residues with their standard equivalents.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    # https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html
    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    if replace_nonstandard_resis:
        fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keep_water)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_pdb, 'w'))


def top_pos_from_sim(simulation):
    """Return (topology, positions) from a running OpenMM Simulation."""
    state = simulation.context.getState(getPositions=True,
                                        enforcePeriodicBox=True)
    return simulation.topology, state.getPositions()


class OMMSetup:
    """
    Helper class to compose an OpenMM simulation object step by step.

    Parameters
    ----------
    structures : list of str
        Paths to prepared PDB files for each component of the system.
        Example: ['lysozyme.pdb']
    nonbonded_cutoff : float
        Non-bonded cutoff distance in nanometers.
    forcefields : list of str
        Force field XML files to load.
    temperature : float
        Simulation temperature in Kelvin.
    pressure : float
        Simulation pressure in bar.
    box_shape : str
        Solvent box shape passed to Modeller.addSolvent (e.g. 'dodecahedron').
    padding : float
        Padding around the solute in nanometers.
    name : str
        Base name used for output files.
    Hmass : float
        Hydrogen mass in atomic mass units (>2 enables longer timesteps via HMR).
    timestep : int
        Integration timestep in femtoseconds.
    """

    def __init__(self,
                 structures,
                 nonbonded_cutoff=1,
                 forcefields=['amber14-all.xml', 'amber14/tip3pfb.xml'],
                 temperature=310.0,
                 pressure=1,
                 box_shape='dodecahedron',
                 padding=1.0,
                 name='system',
                 Hmass=2.0,
                 timestep=2,
                 ):
        from openmm import LangevinMiddleIntegrator, MonteCarloBarostat
        from openmm.unit import nanometer, bar, atomic_mass_unit

        self.structures = structures
        self.nonbonded_cutoff = nonbonded_cutoff * nanometer
        self.integrator_type = LangevinMiddleIntegrator
        self.forcefields = forcefields
        self.temperature = temperature
        self.pressure = pressure * bar
        self.box_shape = box_shape
        self.padding = padding * nanometer
        self.name = name
        self.Hmass = Hmass * atomic_mass_unit
        self.timestep = timestep

    def model(self):
        """Load structures, build the Modeller topology, and add solvent."""
        from openmm.app import PDBFile, Modeller, ForceField
        from openmm.unit import molar

        pdb_file = self.structures[0]
        pdb = PDBFile(pdb_file)
        modeller = Modeller(pdb.topology, pdb.positions)
        if len(self.structures) > 1:
            for structure in self.structures[1:]:
                pdb = PDBFile(structure)
                modeller.add(pdb.topology, pdb.positions)
        self.modeller = modeller
        self.forcefield = ForceField(*self.forcefields)
        self.modeller.addSolvent(self.forcefield, padding=self.padding,
                                 ionicStrength=0.1 * molar, model='tip3p',
                                 boxShape=self.box_shape)

    def make_system(self):
        """Create the OpenMM System object with PME, HBond constraints, and a barostat."""
        from openmm.app import PME, HBonds
        from openmm import MonteCarloBarostat

        system = self.forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=self.nonbonded_cutoff,
            constraints=HBonds,
            hydrogenMass=self.Hmass,
        )
        system.addForce(MonteCarloBarostat(self.pressure, self.temperature))
        self.system = system

    def make_simulation(self):
        """Build the Simulation, set positions, and run energy minimization."""
        from openmm.app import Simulation
        from openmm.unit import picosecond, femtoseconds

        integrator = self.integrator_type(
            self.temperature,
            1 / picosecond,
            self.timestep * femtoseconds,
        )
        simulation = Simulation(self.modeller.topology, self.system, integrator)
        simulation.context.setPositions(self.modeller.positions)
        simulation.minimizeEnergy()
        self.simulation = simulation

    def save(self, output):
        """
        Write the OpenMM system XML and minimized PDB to *output* directory.

        Parameters
        ----------
        output : str
            Destination directory.  Created if it does not exist.
        """
        from openmm import XmlSerializer
        from openmm.app import PDBFile

        os.makedirs(output, exist_ok=True)
        for directory in ['system', 'structures']:
            os.makedirs(f'{output}/{directory}', exist_ok=True)

        topology, positions = top_pos_from_sim(self.simulation)
        with open(f'{output}/system/{self.name}_system.xml', 'w') as outfile:
            outfile.write(XmlSerializer.serialize(self.system))
        with open(f'{output}/structures/{self.name}_minimized.pdb', 'w') as f:
            PDBFile.writeFile(topology, positions, f)

def build_hremd_base_state(
    system_file: str,
    structure_file: str,
    lambda_selection: str,
    temperature: float,
    timestep: float,
):
    """
    Builds the base simulation state and REST-scaled system for HREMD.

    This function reads a system XML and structure PDB, applies femto REST2
    scaling to the specified lambda selection, and generates a thermalized base
    state (positions, velocities, forces). 
    
    It abstracts away the boilerplate required to initialize a `femto.md.hremd`
    compatible system.

    Parameters
    ----------
    system_file : str
        Path to the OpenMM system XML file.
    structure_file : str
        Path to the structure PDB file.
    lambda_selection : str
        MDAnalysis selection string for the atoms to undergo REST2 scaling.
    temperature : float
        The base temperature (K) to thermalize the state to.
    timestep : float
        The integration timestep in femtoseconds.

    Returns
    -------
    tuple
        A tuple containing:
        - system (openmm.System): The REST2-scaled OpenMM system.
        - structure (mdtop.Topology): The system topology.
        - base_state (openmm.State): The fully initialized and thermalized 
          base OpenMM state.
    """
    import femto.md.config
    import femto.md.rest
    import MDAnalysis as mda
    import mdtop
    from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    with open(system_file) as f:
        system = XmlSerializer.deserialize(f.read())

    u = mda.Universe(structure_file)
    solute_idxs = set(u.select_atoms(lambda_selection).atoms.ix)

    rest_config = femto.md.config.REST(scale_torsions=True, scale_nonbonded=True)
    femto.md.rest.apply_rest(system, solute_idxs, rest_config)

    pdb = PDBFile(structure_file)
    structure = mdtop.Topology.from_file(structure_file)

    integrator = LangevinMiddleIntegrator(
        temperature, 1 / unit.picosecond, timestep * unit.femtosecond
    )
    integrator.setRandomNumberSeed(12345)
    
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temperature, 12345)

    base_state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getForces=True,
        getEnergy=True,
        enforcePeriodicBox=True,
    )
    return system, structure, base_state
