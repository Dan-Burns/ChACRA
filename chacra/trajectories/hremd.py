'''
This very accessible implementation of HREMD is thanks to the psivant/femto
package.

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
import pyarrow
from MDAnalysis.analysis.base import AnalysisFromFunction

# if parallel
#import femto.md.utils.mpi
#femto.md.utils.mpi.divide_gpus()

def run_hremd(structure_file, system, temp_min, temp_max, n_systems,
              warmup_steps, steps_per_cycle, cycles, save_interval, 
              checkpoint_interval,
              state=None):
    
    '''
    TODO : MOVE TO SCRIPT
    femto.md.utils.openmm.create_simulation must be passed coords from an
    equilibrated simulation with all the state data.

    state = simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getForces=True,
            getEnergy=True,
            enforcePeriodicBox=True,
        )
    
    '''
    u = mda.Universe(structure_file)
    structure = pmd.load_file(structure_file)

    if type(system) == str:
        with open(system, 'r') as file:
            xml = file.read()

        # Deserialize the XML and create a System object
        system = XmlSerializer.deserialize(xml)
    
        

    rest_config = femto.md.config.REST(scale_torsions=True, scale_nonbonded=True)

    indices = u.select_atoms('protein').atoms.ix # change to omm function
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
        trajectory_interval=save_interval,  # store every n_cycles * n_steps_per_cycle.
        checkpoint_interval=checkpoint_interval
    )
    u_kn, n_k, final_coords = femto.md.hremd.run_hremd(
        simulation,
        states,
        hremd_config,
        # the directory to store sampled reduced potentials and trajectories to
        output_dir=output_dir
    )

    return u_kn, n_k, final_coords

def load_femto_data(hremd_data):
    '''
    hremd_data : str
        Path to the femto state data output (i.e. samples.arrow file).
    '''
    with pyarrow.OSFile(hremd_data, "rb") as file:
        with pyarrow.RecordBatchStreamReader(file) as reader:
            output_table = reader.read_all()
    return output_table.to_pandas()

def get_num_states(df):
    '''
    Get the number of thermodynamic states (same as number of replicas)
    from the femto data.
    '''
    return df['u_kn'][0].shape[0]

def make_combined_traj(structure, traj_dir, n_states):
    '''
    Create an MDAnalysis universe containing all of the replica exchange trajectories.
    Assumes trajectories are named in femto default pattern (r0.dcd, r1.dcd, ...)

    structure : str
        Path to simulation structure file or topology.

    traj_dir : str
        Path to directory containing all of the replica trajectories.

    n_states : int
        The number of replicas in the replica exchange ensemble.

    Returns
    -------
    mda.Universe

    TODO : allow for only specifying structure and traj dir and alternate naming scheme for dcd files.
    '''
    u = mda.Universe(structure, [f"{traj_dir}/r{replica_id}.dcd" for replica_id in range(n_states)])
    return u

def sort_replica_trajectories(df, save_interval, traj_len):
    '''
    Creates a dictionary with state id keys and list of lists values containing each replicas frames 
    that correspond to the state. 

    df : pd.DataFrame
        State data output from femto

    save_interval : int
        The number of cycles that elapse between writing coordinates.
        femto.md.config.HREMD(trajectory_interval=save_interval)

    Returns
    -------
    Dictionary
    keys corresponding to states i to n_states
    values are lists of lists where list index j holds the frames from replica j
    corresponding to state i.
    '''
    
    n_states = df['u_kn'][0].shape[0]
    state_replica_frames = {i:[] for i in range(n_states)}
    # have to know traj_len because there will probably be additional rows in the state_data between the 
    # final saved frame and the next frame that would have been saved.
    replica_to_state_idx = np.vstack((df['replica_to_state_idx'][::save_interval][:traj_len]).values)
    
    for replica in range(n_states):
        frames, states = np.where(replica_to_state_idx == replica)
        for state in range(n_states):
            state_replica_frames[state].append(np.where(states==state)[0])
    
    return state_replica_frames

def write_state_trajectories(structure, traj_dir, state_replica_frames, output_dir, output_selection='protein'):
    '''
    Write separate trajectories for each thermodynamic state. 
    
    Parameters
    ----------
    
    '''
    n_states = len(state_replica_frames)
    u = make_combined_traj(structure, traj_dir, n_states)
    sel = u.select_atoms(output_selection)
    for state in range(n_states):
        state_frames = []
        for replica_id in range(n_states):
            start_frame = replica_id*traj_len
            state_frames.extend(start_frame + state_replica_frames[state][replica_id])
        sel.write(f"{output_dir}/state_{state}.xtc", frames=state_frames)


def get_state_energies(df):
    '''
    df : pd.DataFrame
        State data output from femto.

    Returns
    -------
    np.array of shape n_cycles x n_states
    column i corresponds to the energies sampled at state i
    
    '''
    n_states = df['u_kn'][0].shape[0]
    energies = []
    for row, index in zip(df['u_kn'],df['replica_to_state_idx']):
        energies.append(np.vstack(row)[np.arange(n_states),index])
    return np.vstack(energies)
