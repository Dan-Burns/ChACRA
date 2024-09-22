'''
This very accessible implementation of HREMD is thanks to Simon Boothroyd's 
femto package.

https://github.com/Psivant/femto
https://psivant.github.io/femto/latest/guide-md/

As of 12/26/2023, the fork/branch @ https://github.com/Dan-Burns/femto/tree/enforcePB 
ensures that the trajectory coordinates are properly wrapped to maintain the 
appearance of the box boundaries.
'''
import parmed as pmd
import pathlib
import numpy as np
import openmm.unit
import femto.md.config
import femto.md.constants
import femto.md.hremd
import femto.md.utils.openmm
import femto.md.rest
import pathlib
import femto.md.system
import femto.md.rest
import femto.md.solvate
import openmm.app
from openmm import XmlSerializer
import MDAnalysis as mda

# if parallel
#import femto.md.utils.mpi
#femto.md.utils.mpi.divide_gpus()

def run_hremd(structure_file, system, temp_min, temp_max, n_systems,
              warmup_steps, steps_per_cycle, cycles, save_interval):
    u = mda.Universe(structure_file)
    structure = pmd.load_file(structure_file)

    if type(system) == str:
        with open(system, 'r') as file:
            xml = file.read()

        # Deserialize the XML and create a System object
        system = XmlSerializer.deserialize(xml)
    

    rest_config = femto.md.config.REST(scale_torsions=True, scale_nonbonded=True)

    indices = u.select_atoms('protein').atoms.ix
    solute_idxs = set(indices)

    femto.md.rest.apply_rest(system, solute_idxs, rest_config)


    output_dir = pathlib.Path("hremd-outputs")

    # define the REST2 temperatures to sample at
    temps = list(np.geomspace(temp_min,temp_max,n_systems))
    rest_temperatures = temps * openmm.unit.kelvin
    rest_betas = [
        1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * rest_temperature)
        for rest_temperature in rest_temperatures
    ]

    states = [
        {femto.md.rest.REST_CTX_PARAM: rest_beta / rest_betas[0]}
        for rest_beta in rest_betas
    ]
    # REST requires both beta_m / beta_0 and sqrt(beta_m / beta_0) to be defined
    # we can use a helper to compute the later from the former for each state
    states = [
        femto.md.utils.openmm.evaluate_ctx_parameters(state, system)
        for state in states
    ]

    # create the OpenMM simulation object
    intergrator_config = femto.md.config.LangevinIntegrator(
        timestep=2.0 * openmm.unit.femtosecond,
    )
    integrator = femto.md.utils.openmm.create_integrator(
        intergrator_config, rest_temperatures[0]
    )

    simulation = femto.md.utils.openmm.create_simulation(
        system,
        structure,
        coords=None,  # or None to use the coordinates / box in structure
        integrator=integrator,
        state=states[0],
        platform=femto.md.constants.OpenMMPlatform.CUDA,
    )

    # define how the HREMD should be run
    hremd_config = femto.md.config.HREMD(
        # the number of steps to run each replica for before starting to
        # propose swaps
        n_warmup_steps=warmup_steps,
        # the number of steps to run before proposing swaps
        n_steps_per_cycle=steps_per_cycle,
        # the number of 'swaps' to propose - the total simulation length
        # will be n_warmup_steps + n_steps * n_cycles
        n_cycles=cycles,
        # the frequency with which to store trajectories of each replica.
        # set to None to not store trajectories
        trajectory_interval=save_interval  # store every n_cycles * n_steps_per_cycle.
    )
    femto.md.hremd.run_hremd(
        simulation,
        states,
        hremd_config,
        # the directory to store sampled reduced potentials and trajectories to
        output_dir=output_dir
    )
