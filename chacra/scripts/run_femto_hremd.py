
import argparse

import femto.md.utils.mpi
import MDAnalysis as mda
import mdtop
import numpy as np
import openmm
from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
from openmm.app import PDBFile, Simulation

femto.md.utils.mpi.divide_gpus()
import pathlib
from datetime import datetime

import femto.md.config
import femto.md.constants
import femto.md.hremd
import femto.md.rest
import femto.md.simulate
import femto.md.utils.openmm
import openmm.unit

parser = argparse.ArgumentParser(description="Run HREMD simulation.")

# Add arguments for system file and structure
parser.add_argument(
    "--system_file",
    type=str,
    required=True,
    help="Path to the system XML file.",
)
parser.add_argument(
    "--structure_file",
    type=str,
    required=True,
    help="Path to the structure PDB file.",
)
parser.add_argument(
    "--n_cycles",
    type=int,
    default=10000,
    help="The number of replica_exchange attempts for this run.",
)
parser.add_argument(
    "--steps_per_cycle",
    type=int,
    default=1000,
    help="The number of timesteps between replica exchange attempts.",
)
parser.add_argument(
    "--min_temp",
    type=float,
    default=290,
    help="""
        The minimum effective temperature (in kelvin) of the replica
        exchange ensemble.This is the same as the solvent temperature for
        all replicas.
        """,
)
parser.add_argument(
    "--max_temp",
    type=float,
    default=450,
    help="""
        The maximum effective temperature (in kelvin) of the replica 
        exchange ensemble.
        """,
)
parser.add_argument(
    "--n_systems", type=int, required=True, help="The number of replicas."
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=10,
    help="Save trajectory data at this cycle interval.",
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=100,
    help="Save checkpoints at this cycle interval.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=0,
    help="""
        The number of warmup steps to run before starting replica exchange 
        attempts. Only necessary on the first run.
        """,
)
parser.add_argument(
    "--lambda_selection",
    type=str,
    default="protein",
    help="""
    The MDAnalysis selection to which the lambda scaling will be applied.
    """,
)
args = parser.parse_args()
system_file = args.system_file
with open(system_file, "r") as file:
    xml = file.read()
system = XmlSerializer.deserialize(xml)
structure_file = args.structure_file

u = mda.Universe(structure_file)
indices = u.select_atoms(
    args.lambda_selection
).atoms.ix  # change to omm function
solute_idxs = set(indices)
temp_min = args.min_temp
temp_max = args.max_temp
n_systems = args.n_systems
warmup_steps = args.warmup_steps
steps_per_cycle = args.steps_per_cycle
cycles = args.n_cycles
save_interval = args.save_interval
checkpoint_interval = args.checkpoint_interval

rest_config = femto.md.config.REST(
    scale_torsions=True, scale_nonbonded=True
)
femto.md.rest.apply_rest(system, solute_idxs, rest_config)
pdb = PDBFile(structure_file)
structure = mdtop.Topology.from_file(structure_file)
integrator = LangevinMiddleIntegrator(
    290, 1 / unit.picosecond, 2 * unit.femtosecond
)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    getForces=True,
    getEnergy=True,
    enforcePeriodicBox=True,
)
# simulation.minimizeEnergy()

output_dir = pathlib.Path("hremd-outputs")

# define the REST2 temperatures to sample at
temps = list(np.geomspace(temp_min, temp_max, n_systems))
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
    # will be n_warmup_steps + n_steps * n_cycles
    n_cycles=cycles,
    # the frequency with which to store trajectories of each replica.
    # set to None to not store trajectories
    trajectory_interval=save_interval,  # store every 10 * 500 steps.
    checkpoint_interval=checkpoint_interval,
)
print(datetime.now().strftime("%H:%M"))
femto.md.hremd.run_hremd(
    simulation,
    states,
    hremd_config,
    # the directory to store sampled reduced potentials and trajectories to
    output_dir=output_dir,
)

print(datetime.now().strftime("%H:%M"))
