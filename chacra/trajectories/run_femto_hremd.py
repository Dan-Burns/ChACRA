import argparse
import femto.md.utils.mpi
import openmm
from openmm import *
from openmm.app import *
import MDAnalysis as mda
import numpy as np
import parmed as pmd
import pathlib
from datetime import datetime
import openmm.unit
import femto.md.simulate
import femto.md.config
import femto.md.constants
import femto.md.hremd
import femto.md.utils.openmm
import femto.md.rest

'''
Run a Hamiltonian Replica Exchange Molecular Dynamics (HREMD) simulation using femto / openmm.
If continuing from a checkpoint, the checkpoint file must be in output_path (default is 'hremd-outputs').
https://github.com/Psivant/femto
https://psivant.github.io/femto/latest/guide-md/



'''
femto.md.utils.mpi.divide_gpus()

parser = argparse.ArgumentParser(description='Run HREMD simulation.')

parser.add_argument('--system_file', type=str, required=True, 
                    help='Path to the system XML file.')
parser.add_argument('--structure_file', type=str, required=True, 
                    help='Path to the structure PDB file. If this is the beginning of the HREMD simulation, it is '\
                    'necessary to provide energy minimized coordinates even if the structure has been equilibrated.')
parser.add_argument('--lambda_selection', type=str, default='protein', help='Provide an MDAnalysis selection string '\
                    'defining the atoms that Hamiltonian scaling will be applied to.' )
parser.add_argument('--temp_min', type=float, default=290, 
                    help='The minimum effective temperature that the replicas will sample in kelvin. '\
                    'This will be the temperature of that the non-scaled portion of the system is running at.' )
parser.add_argument('--temp_max', type=float, 
                    help='The maximum effective temperature that the replicas will sample in kelvin.')
parser.add_argument('--n_systems', type=int, default=20, 
                    help='The number of systems in the replica exchange ensemble.')
parser.add_argument('--warmup_steps', type=int, default=0, 
                    help='The number of simulation steps to perform before beginning replica exchange attempts. Serves to equilibrate the '\
                    'systems to their effective temperature and decorrelate the structures. Will not be performed if checkpoint file is present.')
parser.add_argument('--steps_per_cycle', type=int, default=1000, 
                    help='The number of steps to perform between replica exchange attempts.')
parser.add_argument('--n_cycles', type=int, default=10000, 
                    help='The number of replica exchange attempts to perform. Total steps per replica will be steps_per_cycle*n_cycles.')
parser.add_argument('--save_interval', type=int, default=10, 
                    help='The number of replica exchange attempts between saving trajectory data. State data is saved every cycle by default. '\
                    'Steps between trajectory frames will be equal to save_interval*steps_per_cycle')
parser.add_argument('--checkpoint_interval', type=int, default=100,
                    help='The number of cycles in between updating the checkpoint files.')
parser.add_argument('--output_path', type=str, default='hremd-outputs', 
                    help='The path to the directory where the data will be saved.')

args = parser.parse_args()

# system
system_file = args.system_file
with open(system_file, 'r') as file:
        xml = file.read()
system = XmlSerializer.deserialize(xml)

structure_file = args.structure_file
lambda_selection = args.lambda_selection
temp_min = args.temp_min
temp_max = args.temp_max
n_systems = args.n_systems
warmup_steps = args.warmup_steps
steps_per_cycle = args.steps_per_cycle
cycles = args.n_cycles
save_interval = args.save_interval
checkpoint_interval = args.checkpoint_interval
output_path = args.output_path


pdb = PDBFile(structure_file)
structure = pmd.load_file(structure_file)
integrator = LangevinMiddleIntegrator(temp_min,1/unit.picosecond, 2*unit.femtosecond)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

u = mda.Universe(structure_file)
indices = u.select_atoms(lambda_selection).atoms.ix 
solute_idxs = set(indices)


state = simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getForces=True,
            getEnergy=True,
            enforcePeriodicBox=True,
        )

rest_config = femto.md.config.REST(scale_torsions=True, scale_nonbonded=True)
femto.md.rest.apply_rest(system, solute_idxs, rest_config)
output_dir = pathlib.Path(output_path)

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
# we can use a helper to compute the latter from the former for each state
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
    coords=state,  # or None to use the coordinates / box in structure
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
    n_cycles=cycles,
    # the frequency with which to store trajectories of each replica.
    # set to None to not store trajectories
    trajectory_interval=save_interval,  
    checkpoint_interval=checkpoint_interval
)
print(datetime.now().strftime("%H:%M"))
femto.md.hremd.run_hremd(
    simulation,
    states,
    hremd_config,
    # the directory to store sampled reduced potentials and trajectories to
    output_dir=output_dir
)

print(datetime.now().strftime("%H:%M"))