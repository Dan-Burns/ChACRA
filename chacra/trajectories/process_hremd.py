import numpy as np
import MDAnalysis as mda
import pyarrow




















# Functions to process the HREMD Output
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

def write_state_trajectories(structure, traj_dir, state_replica_frames, traj_len, output_dir, output_selection='protein'):
    '''
    Write separate trajectories for each thermodynamic state from a femto HREMD simulation. 
    
    Parameters
    ----------
    structure : str
        Path to simulation topology / pdb.
    
    traj_dir : str
        Path to directory with the femto hremd trajectories.

    state_replica_frames : dict
        The dictionary produced by sort_replica_trajectories(). Contains keys of state ids
        and lists of lists that hold the frames of each replica (replica i = list index i) corresponding to that state.

    traj_len : int
        The number of frames contained in a single replica trajectory.
        Load any of the trajectories and get len(u.trajectory).

    output_dir : str
        Path to the directory where the individual state trajectories will be written.

    output_selection : str
        An MDAnalysis selection of the atoms that should be written out for the individual state trajectories.

    Returns
    -------
    None. Writes trajectories to output_dir.
    
    
    '''
    n_states = len(state_replica_frames)
    # all the replicas in one trajectory
    u = make_combined_traj(structure, traj_dir, n_states)
    sel = u.select_atoms(output_selection)

    for state in range(n_states):
        # the frame indices from all the replicas corresponding to one state 
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
        # each row contains n_states arrays of arrays of size n_states
        # the index contains the replica id in the element corresponding to the state it's being simulated at
        # at that frame.
        # by indexing the vstacked row with [np.arange(n_states), index], you retrieve the energy of the 
        # replica being simulated at that state
        energies.append(np.vstack(row)[np.arange(n_states),index])
    return np.vstack(energies)
