import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import align
import pyarrow
import os
from ..utils import get_resources
from multiprocessing import Pool
from pathlib import Path
import sys

#sys.getsizeof()

# Functions to process the HREMD Output
def load_femto_data(hremd_data:str|os.PathLike)-> pd.DataFrame:
    '''
    Load femto HREMD state data output (.arrow) file into a pandas DataFrame.

    Parameters
    ----------
    hremd_data : str
        Path to the femto state data output (i.e. samples.arrow file).
    '''
    with pyarrow.OSFile(str(hremd_data), "rb") as file:
        with pyarrow.RecordBatchStreamReader(file) as reader:
            output_table = reader.read_all()
    return output_table.to_pandas()

def get_num_states(df:pd.DataFrame) -> int:
    '''
    Get the number of thermodynamic states (same as number of replicas)
    from the femto data.

    Parameters
    ----------
    df : pd.DataFrame
        State data output from femto.
    Returns
    -------
    int
    The number of thermodynamic states in the HREMD simulation.
    '''
    return df['u_kn'][0].shape[0]

def make_combined_traj(structure:str|os.PathLike, 
                       traj_dir:str|os.PathLike, 
                       n_states:int) -> mda.Universe:
    '''
    Create an MDAnalysis universe containing all of the replica exchange 
    trajectories. Assumes trajectories are named in femto default pattern 
    (i.e. r0.dcd, r1.dcd, ...)

    Parameters
    ----------
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
    u = mda.Universe(str(structure), [str(Path(traj_dir)/f"r{replica_id}.dcd") 
                                 for replica_id in range(n_states)])
    return u


def sort_replica_trajectories(df:pd.DataFrame, 
                              save_interval:int, 
                              traj_len:int) -> dict:
    '''
    Creates a dictionary with state id keys and list of lists containing 
    each replica's frames that correspond to the state. 

    Parameters
    ----------
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
    keys corresponding to states i to n_states. values are lists of lists where 
    list index j holds the frames from replica j corresponding to state i.
    '''
    
    n_states = df['u_kn'][0].shape[0]
    replica_state_frames = {i:[] for i in range(n_states)}
    # have to know traj_len because there could be additional rows in the 
    # state_data between the final saved frame and the next frame that would have 
    # been saved.
    replica_to_state_idx = np.vstack(
                (df['replica_to_state_idx'][::save_interval]).values[:traj_len])
    
   
    for replica in range(n_states):
        for state in range(n_states):
            frames = np.where(replica_to_state_idx[:, replica] == state)[0]
            replica_state_frames[replica].append(frames)

    return replica_state_frames

# keeping for posterity. Doesn't work but was fast...
# def write_state_trajectories_test(structure, traj_dir, hremd_data, save_interval, output_dir, output_selection='protein'):
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

def get_state_coordinates_from_replica(structure:str|os.PathLike, 
                             traj_dir:str|os.PathLike, 
                             hremd_data:str|os.PathLike|dict, 
                             save_interval:int,  
                             selection:str='protein',
                             replica_index:int=None,
                             state_index:int=None) -> np.ndarray:
    '''
    Get the frames from all replicas that correspond to a specific state.

    Parameters
    ----------
    structure : str
        Path to simulation topology / pdb.

    traj_dir : str
        Path to directory with the femto hremd trajectories.

    hremd_data : str
        Path to the femto state data output (i.e. samples.arrow file).
        Or the state_replica_frames dictionary.

    save_interval : int
        The number of cycles that elapse between writing coordinates.

    selection : str, optional
        An MDAnalysis selection of the atoms that should be used. Default is 'protein'.

    replica_index : int
        Index of the replica to get frames from.

    state_index : int, optional
        Index of the state to get frames for.

    Returns
    -------
    np.ndarray
        Stacked array of frames from all replicas that correspond to the specified state.
    '''
    structure = str(structure)
    rep = replica_index
    u = mda.Universe(structure, str(Path(traj_dir)/f"r{rep}.dcd"))
    traj_len = len(u.trajectory)
    sel = u.select_atoms(selection)
    n_atoms = sel.n_atoms
    if isinstance(hremd_data, (str, os.PathLike)):
        df = load_femto_data(hremd_data)
        state_replica_frames = sort_replica_trajectories(df, save_interval, traj_len)
    else:
        state_replica_frames = hremd_data

    state = state_index 
    state_traj_len = len(state_replica_frames[rep][state])

    if state_traj_len == 0:
        return None
    else:
        coordinates = np.zeros((state_traj_len,n_atoms,3))
    
        for i,frame in enumerate(state_replica_frames[rep][state]):
            u.trajectory[frame]
            coordinates[i] = sel.positions 
        
        return coordinates

    
    
    
    
def write_state_trajectory(structure:str|os.PathLike, 
                             traj_dir:str|os.PathLike, 
                             hremd_data:str|os.PathLike|dict, 
                             save_interval:int,  
                             output_dir:str, 
                             selection:str='protein',
                             ref=None,
                             state_index=None):
    '''
    Separate the state trajectory for a thermodynamic state from the femto HREMD
    replica trajectories.
    
    Parameters
    ----------
    structure : str
        Path to simulation topology / pdb.

    traj_dir : str
        Path to directory with the femto hremd trajectories.

    hremd_data : str
        Path to the femto state data output (i.e. samples.arrow file).

    save_interval : int
        The number of cycles that elapse between writing coordinates.

    output_dir : str
        Path to the directory where the individual state trajectories will be written.

    selection : str, optional
        An MDAnalysis selection of the atoms that should be used. Default is 'protein'.

    ref : str, optional
        Path to reference structure. If ref is provided, coordinates are aligned to 
        C-alphas of selection. Default is None.

    state_index : int
        Index of the state to write. 

    Returns
    -------
    Writes state trajectory for state_index to output_dir.
    
    '''
    structure = str(structure)
    ref = str(ref) if ref is not None else None
    traj = [file for file in os.listdir(traj_dir) if (file.endswith("dcd"))][0]
    u = mda.Universe(structure, str(Path(traj_dir)/traj))
    traj_len = len(u.trajectory)

    if isinstance(hremd_data, (str, os.PathLike)):
        df = load_femto_data(hremd_data)
        state_replica_frames = sort_replica_trajectories(df, save_interval, traj_len)
    else:
        state_replica_frames = hremd_data
    n_states = len(state_replica_frames)

    if ref is not None:
        ref = mda.Universe(ref)
    else:
        ref = mda.Universe(structure)

    state = state_index

    out_u = mda.Universe(structure)
    sel = out_u.select_atoms(selection)

    coords = [get_state_coordinates_from_replica(structure, 
                             traj_dir, 
                             state_replica_frames,
                             save_interval,  
                             selection='protein',
                             replica_index=i,
                             state_index=state)

                             for i in range(n_states)
                             ]
    coords = [c for c in coords if c is not None]
    coordinates = np.concatenate(coords, axis=0)
    # These coordinates will not match write_state_trajectories because they are 
    # added sequentially per replica and not into the array index corresponding to 
    # the frame they were taken from. If contact frequencies match - it's correct.

    with mda.Writer(str(Path(output_dir)/f"state_{state}.xtc"), n_atoms=sel.n_atoms) as writer:
        for i in range(coordinates.shape[0]):
            sel.positions = coordinates[i]  # Set the positions to the NumPy array
            align.alignto(sel, ref, select=f"({selection}) and name CA")
            writer.write(sel.atoms) 

def write_state_trajectory_parallel(structure:str|os.PathLike, 
                             traj_dir:str|os.PathLike, 
                             hremd_data:str|os.PathLike|dict, 
                             save_interval:int,  
                             output_dir:str|os.PathLike, 
                             selection:str='protein',
                             ref=None,
                             state_index=None,
                             n_jobs=None):
    
    '''
    Separate the state trajectory for a thermodynamic state from the femto HREMD
    replica trajectories. Each replica is assigned to a single separate process.
    
    Parameters
    ----------
    structure : str
        Path to simulation topology / pdb.

    traj_dir : str
        Path to directory with the femto hremd trajectories.

    hremd_data : str
        Path to the femto state data output (i.e. samples.arrow file).

    save_interval : int
        The number of cycles that elapse between writing coordinates.

    output_dir : str
        Path to the directory where the individual state trajectories will be written.

    selection : str, optional
        An MDAnalysis selection of the atoms that should be used. Default is 'protein'.

    ref : str, optional
        Path to reference structure. If ref is provided, coordinates are aligned to 
        C-alphas of selection. Default is None.

    state_index : int
        Index of the state to write. 

    Returns
    -------
    Writes state trajectory for state_index to output_dir.
    
    '''
    structure = str(structure)
    ref = str(ref) if ref is not None else None
    traj = [file for file in os.listdir(traj_dir) if (file.endswith("dcd"))][0]
    u = mda.Universe(structure, Path(traj_dir)/traj)
    traj_len = len(u.trajectory)

    
    if isinstance(hremd_data, (str, os.PathLike)):
        df = load_femto_data(hremd_data)
        state_replica_frames = sort_replica_trajectories(df, save_interval, traj_len)
    else:
        state_replica_frames = hremd_data
    n_replicas = len(state_replica_frames)

    if ref is not None:
        ref = mda.Universe(ref)
    else:
        ref = mda.Universe(structure)

    state = state_index

    out_u = mda.Universe(structure)
    sel = out_u.select_atoms(selection)

    coords_args_list = [(structure, 
                        traj_dir,
                        hremd_data,
                        save_interval, 
                        selection,
                        i, 
                        state)
                        for i in range(n_replicas)]
    if n_jobs is None:
        resources = get_resources()
        n_jobs = resources['num_cores']
    with Pool(n_jobs) as worker_pool:
        coords = worker_pool.starmap(get_state_coordinates_from_replica,
                                     coords_args_list)

    coords = [c for c in coords if c is not None]
    coordinates = np.concatenate(coords, axis=0)
    # These coordinates will not match write_state_trajectories because they are 
    # added sequentially per replica and not into the array index corresponding to 
    # the frame they were taken from. If contact frequencies match - it's correct.

    with mda.Writer(str(Path(output_dir)/f"state_{state}.xtc"), n_atoms=sel.n_atoms) as writer:
        for i in range(coordinates.shape[0]):
            sel.positions = coordinates[i]  # Set the positions to the NumPy array
            align.alignto(sel, ref, select=f"({selection}) and name CA")
            writer.write(sel.atoms) 


def write_state_trajectories_parallel(structure:str|os.PathLike, 
                             traj_dir:str|os.PathLike, 
                             hremd_data:str|os.PathLike|dict, 
                             save_interval:int, 
                             output_dir:str|os.PathLike, 
                             selection:str='protein',
                             ref=None,
                             n_jobs=None
                             ):
    
    '''
    Write separate trajectories for each thermodynamic state from a femto HREMD simulation.
    Does it in the order of the frames of all the combined trajectories ... which doesn't really matter.

    ref : str
        Path to reference structure. If ref is provided, coordinates are aligned to 
        C-alphas of selection.
    '''
    structure = str(structure)
    ref = str(ref) if ref is not None else None
    traj = [file for file in os.listdir(traj_dir) if (file.endswith("dcd"))][0]# taking the first trajectory in the list to get the individual traj_len
    u = mda.Universe(structure, Path(traj_dir)/traj)
    traj_len = len(u.trajectory)
    
    if isinstance(hremd_data, (str, os.PathLike)):
        df = load_femto_data(hremd_data)
        state_replica_frames = sort_replica_trajectories(df, save_interval, traj_len)
    else:
        state_replica_frames = hremd_data


    n_states = len(state_replica_frames)

    state_args_list = [(structure, 
                        traj_dir, 
                        state_replica_frames,
                        save_interval, 
                        output_dir,
                        selection,
                        ref,
                        i) 
                        for i in range(n_states)] 
    if n_jobs is None:
        resources = get_resources()
        n_jobs = resources['num_cores']                
    with Pool(n_jobs) as worker_pool:
        worker_pool.starmap(write_state_trajectory,
                                     state_args_list)
    





def write_state_trajectories(structure, 
                             traj_dir, 
                             hremd_data, 
                             save_interval, 
                             output_dir, 
                             selection='protein',
                             ref=None,
                             ):
    
    '''
    Will be deprecated in favor of write_state_trajectories_parallel.
    Write separate trajectories for each thermodynamic state from a femto HREMD simulation.
    Does it in the order of the frames of all the combined trajectories ... which doesn't really matter.

    ref : str
        Path to reference structure. If ref is provided, coordinates are aligned to 
        C-alphas of selection.
    '''

    traj = [file for file in os.listdir(traj_dir) if (file.endswith("dcd"))][0]# taking the first trajectory in the list to get the individual traj_len
    u = mda.Universe(structure, Path(traj_dir)/traj)
    traj_len = len(u.trajectory)
    sel = u.select_atoms(selection)
    n_atoms = sel.n_atoms
    df = load_femto_data(hremd_data)
    state_replica_frames = sort_replica_trajectories(df, save_interval, traj_len)
    n_states = len(state_replica_frames)
    if ref is not None:
        ref = mda.Universe(ref)
    else:
        ref = mda.Universe(structure)

    
    for state in range(n_states):
        coordinates = np.zeros((traj_len,n_atoms,3))
        for rep in range(n_states):
            u = mda.Universe(structure, Path(traj_dir)/f"r{rep}.dcd")
            sel = u.select_atoms(selection)
            for frame in state_replica_frames[rep][state]:
                u.trajectory[frame] # Move to the specified frame

                coordinates[frame] = sel.positions # each frame is unique to a given state, so these array indices aren't filled sequentially.
                # yet they do correspond to the correct relative timepoints once they're all in place.
        out_u = mda.Universe(structure)
        sel = out_u.select_atoms(selection)
        with mda.Writer(Path(output_dir)/f"state_{state}.xtc", n_atoms=sel.n_atoms) as writer:
            for i in range(coordinates.shape[0]):
                sel.positions = coordinates[i]  # Set the positions to the NumPy array
                align.alignto(sel, ref, select=f"({selection}) and name CA")
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

def get_exchange_probability(df, state_i, state_j):
    '''
    df : pd.DataFrame
        State data output from femto

    Returns
    -------
    float
    The probability of exchange between states i and j
    '''
    array = np.vstack(df['n_accepted_swaps'].values)
    attempt_array = np.vstack(df['n_proposed_swaps'].values)
    final_swap = array[-1]
    final_attempts = attempt_array[-1]
    


    return final_swap[state_i][state_j]/final_attempts[state_i][state_j]

def get_exchange_probabilities(data):
    '''
    Get all the pairwise (nearest neighbors only) exchange
    probabilities

    Parameters
    ----------
    df : pd.DataFrame | str
        State data output from femto loaded in dataframe
        or path to the samples.arrow file.

    Returns
    -------
    np.ndarray 
    '''
    if type(data) == str:
        df = load_femto_data(data)
    else:
        df = data
    array = np.vstack(df['n_accepted_swaps'].values)
    attempt_array = np.vstack(df['n_proposed_swaps'].values)
    n_states = array.shape[1]
    swaps = np.vstack(array[-1])
    attempts = np.vstack(attempt_array[-1])
    states_i = np.array(range(n_states-1))
    states_j = np.array(range(1,n_states))
    indices = [i for i in zip(states_i, states_j)] # can return indices if it's necessary
    return swaps[states_i,states_j]/attempts[states_i, states_j]

def concatenate_runs(state_trajectory_dir):
    '''
    Concatentate the trajectories from multiple runs for each state.
    '''
    return

def freq_frames(freq_file):
    '''
    Return the number of frames that a contact frequency file 
    was calculated for.

    freq_file : str
        Path to getcontact contact frequency file.

    Returns
    -------
    int
    The number of frames the contact frequencies were calculated for.
    '''
    with open(freq_file, 'r') as f:
        frames = f.readline().strip().split()[1]
    frames = int(frames.split(":")[-1])
    return frames

