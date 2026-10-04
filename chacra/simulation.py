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
    forcefield : str
        Forcefield(s) (comma separated) to use for the simulation.
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


def _run_sim_worker(builder_func, steps, kwargs):
    """Worker process that builds and runs the simulation."""
    import openmm.unit
    
    # Build the simulation fresh inside this process
    simulation = builder_func(**kwargs)
    
    # Run the simulation
    simulation.step(steps)
    
    # Return the final state potential energy
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit_system(openmm.unit.md_unit_system)


def run_concurrently_with_mps(builder_func, configs, steps=1000):
    """
    Run multiple OpenMM simulations concurrently on a single GPU using CUDA MPS.
    
    This function delegates the OpenMM initialization to child processes to bypass
    OpenMM's inability to pickle and share Simulation objects across processes. 
    It automatically configures the MPS environment for the workers.
    
    Parameters
    ----------
    builder_func : callable
        A function that takes **kwargs and returns an openmm.app.Simulation object.
        This function will be executed in the child processes.
    configs : list of dict
        A list of kwargs dictionaries to pass to builder_func for each replica.
    steps : int
        Number of steps to run for each simulation.
        
    Returns
    -------
    list of float
        The final potential energy of each simulation.

    Example
    -------
    >>> def my_sim_builder(temperature):
    ...     # This runs inside the child process!
    ...     from chacra.simulation import build_hremd_base_state
    ...     from openmm.app import Simulation
    ...     from openmm import LangevinMiddleIntegrator, unit
    ...     
    ...     system, structure, base_state = build_hremd_base_state(
    ...         system_file="system.xml", structure_file="structure.pdb", 
    ...         lambda_selection="protein", temperature=290.0, timestep=2.0
    ...     )
    ...     
    ...     integrator = LangevinMiddleIntegrator(
    ...         temperature * unit.kelvin, 1/unit.picosecond, 2.0*unit.femtosecond
    ...     )
    ...     sim = Simulation(structure.topology, system, integrator)
    ...     sim.context.setState(base_state)
    ...     return sim
    ...
    >>> configs = [
    ...     {"temperature": 290.0},
    ...     {"temperature": 310.0},
    ... ]
    >>> energies = run_concurrently_with_mps(my_sim_builder, configs, steps=5000)
    """
    import multiprocessing as mp
    import os
    import time
    
    num_replicas = len(configs)
    
    # 1. Configure the environment for MPS
    # Tell MPS to evenly divide the GPU threads among the replicas
    thread_pct = max(1, 200 // num_replicas)
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(thread_pct)
    
    print(f"Starting {num_replicas} concurrent simulations (MPS thread %: {thread_pct})")
    
    # 2. Use multiprocessing "spawn" to ensure completely clean CUDA contexts
    ctx = mp.get_context("spawn")
    
    start_time = time.time()
    
    # 3. Launch the workers
    with ctx.Pool(processes=num_replicas) as pool:
        # We pass the builder_func, the number of steps, and the specific config dict
        results = pool.starmap(
            _run_sim_worker, 
            [(builder_func, steps, cfg) for cfg in configs]
        )
        
    wall_time = time.time() - start_time
    print(f"All simulations finished in {wall_time:.2f} seconds.")
    
    return results
