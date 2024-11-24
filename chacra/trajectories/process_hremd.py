import numpy as np
import MDAnalysis as mda
import pyarrow
import os


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

    traj_len : int
        The length of an individual replica trajectory.

    Returns
    -------
    Dictionary
    keys corresponding to states i to n_states
    values are lists of lists where list index j holds the frames from replica j
    corresponding to state i.
    '''
    
    n_states = df['u_kn'][0].shape[0]
    replica_state_frames = {i:[] for i in range(n_states)}
    # have to know traj_len because there could be additional rows in the state_data between the 
    # final saved frame and the next frame that would have been saved.
    replica_to_state_idx = np.vstack((df['replica_to_state_idx'][::save_interval]).values[:traj_len])
    
   
    for replica in range(n_states):
        for state in range(n_states):
            frames = np.where(replica_to_state_idx[:, replica] == state)[0]
            replica_state_frames[replica].append(frames)

    return replica_state_frames

# This is way faster than the following version
# But it doesn't write the correct state trajectories... something about indexing the frames by adding replica_id*traj_len doesn't work.
# def write_state_trajectories(structure, traj_dir, hremd_data, save_interval, output_dir, output_selection='protein'):
#     '''
#     Write separate trajectories for each thermodynamic state from a femto HREMD simulation. 
    
#     Parameters
#     ----------
#     structure : str
#         Path to simulation topology / pdb.
    
#     traj_dir : str
#         Path to directory with the femto hremd trajectories.

#      hremd_data : str
#         Path to the femto state data output (i.e. samples.arrow file).
    
#     save_interval : int
#         The number of cycles that elapse between writing coordinates.
#         femto.md.config.HREMD(trajectory_interval=save_interval)

#     output_dir : str
#         Path to the directory where the individual state trajectories will be written.

#     output_selection : str
#         An MDAnalysis selection of the atoms that should be written out for the individual state trajectories.

#     Returns
#     -------
#     Writes trajectories to output_dir.
    
   
#     '''
    
#     traj = [file for file in os.listdir(traj_dir) if (file.endswith("dcd"))][0]# taking the first trajectory in the list to get the individual traj_len
#     traj_len = len(mda.Universe(structure, f"{traj_dir}/{traj}").trajectory)
#     df = load_femto_data(hremd_data)
#     state_replica_frames = alt_sort_replica_trajectories(df, save_interval, traj_len)
#     n_states = len(state_replica_frames)
   
#     # all the replicas in one trajectory
#     u = make_combined_traj(structure, traj_dir, n_states)
#     sel = u.select_atoms(output_selection)
#     os.makedirs(output_dir, exist_ok=True)

#     for state in range(n_states):
#         # the frame indices from all the replicas corresponding to one state 
#         state_frames = []

#         for replica_id in range(n_states):
#             start_frame = replica_id*traj_len # move to the first frame of replica replica_id in the combined trajectory
#             state_frames.extend(start_frame + state_replica_frames[replica_id][state])
#         with mda.Writer(f"{output_dir}/state_{state}.xtc", sel.n_atoms) as w:
#             for ts in u.trajectory[state_frames]:
#                 w.write(sel)
#         ##sel.write(f"{output_dir}/state_{state}.xtc", frames=u.trajectory[state_frames])



# alternative - memory intensive and slow but works

def write_state_trajectories(structure, traj_dir, hremd_data, save_interval, output_dir, selection='protein'):

    traj = [file for file in os.listdir(traj_dir) if (file.endswith("dcd"))][0]# taking the first trajectory in the list to get the individual traj_len
    u = mda.Universe(structure, f"{traj_dir}/{traj}")
    traj_len = len(u.trajectory)
    sel = u.select_atoms(selection)
    n_atoms = sel.n_atoms
    df = load_femto_data(hremd_data)
    state_replica_frames = sort_replica_trajectories(df, save_interval, traj_len)
    n_states = len(state_replica_frames)
    
    for state in range(n_states):
        coordinates = np.zeros((traj_len,n_atoms,3))
        for rep in range(n_states):
            u = mda.Universe(structure, f"{traj_dir}/r{rep}.dcd")
            sel = u.select_atoms(selection)
            for frame in state_replica_frames[rep][state]:
                u.trajectory[frame] # Move to the specified frame

                coordinates[frame] = sel.positions # Extract coordinates
        out_u = mda.Universe(structure)
        sel = out_u.select_atoms(selection)
        with mda.Writer(f"{output_dir}/state_{state}.xtc", n_atoms=sel.n_atoms) as writer:
            for i in range(coordinates.shape[0]):
                sel.positions = coordinates[i]  # Set the positions to the NumPy array
                writer.write(sel.atoms) 

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
        # each row contains n_states arrays of size n_states
        # the index contains the replica id in the element corresponding to the state it's being simulated at
        # at that frame.
        # by indexing the vstacked row with [np.arange(n_states), index], you retrieve the energy of the 
        # replica being simulated at that state
        energies.append(np.vstack(row)[np.arange(n_states),index])
    return np.vstack(energies)

def concatenate_runs(state_trajectory_dir):
    '''
    Concatentate the trajectories from multiple runs for each state.
    '''
    return