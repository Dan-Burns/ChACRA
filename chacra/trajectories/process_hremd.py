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


def get_traj_size(structure:str|os.PathLike,
                       traj:str|os.PathLike,
                       selection:str=None)-> int:
    
    '''
    Get the size of the trajectory file in bytes.
    Parameters
    ----------
    structure : str|os.PathLike
        Path to simulation topology / pdb.
    traj : str|os.PathLike
        Path to the trajectory file (i.e. r0.dcd).
    Returns
    -------
    int
        Size of the trajectory file in bytes.
    '''
    structure = str(structure)
    traj = str(traj)
    u = mda.Universe(structure, traj)   
    n_frames = len(u.trajectory)
    if selection is not None:
        coords = u.select_atoms(selection).positions
    else:
        coords = u.atoms.positions
    size = sys.getsizeof(coords) # size of the coordinates array in bytes
    return size*n_frames

def determine_max_jobs(mem_per_job_gb: float,
                       available_mem_gb: float, 
                       reserve_pct: float = 0.2) -> int:
    usable_mem = available_mem_gb * (1 - reserve_pct)
    return max(1, int(usable_mem // mem_per_job_gb))

################# Functions called from the ReplicaHandler class ###############


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
        An MDAnalysis selection of the atoms that should be used. Default is 
        'protein'.

    replica_index : int
        Index of the replica to get frames from.

    state_index : int, optional
        Index of the state to get frames for.

    Returns
    -------
    np.ndarray
        Stacked array of frames from all replicas that correspond to the 
        specified state.
    '''
    structure = str(structure)
    rep = replica_index
    u = mda.Universe(structure, str(Path(traj_dir)/f"r{rep}.dcd"))
    traj_len = len(u.trajectory)
    sel = u.select_atoms(selection)
    n_atoms = sel.n_atoms
    if isinstance(hremd_data, (str, os.PathLike)):
        df = load_femto_data(hremd_data)
        state_replica_frames = sort_replica_trajectories(df, 
                                                        save_interval, traj_len)
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
        Path to the directory where the individual state trajectories will be 
        written.

    selection : str, optional
        An MDAnalysis selection of the atoms that should be used. Default is 
        'protein'.

    ref : str, optional
        Path to reference structure. If ref is provided, coordinates are aligned
        to C-alphas of selection. Default is None.

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
        state_replica_frames = sort_replica_trajectories(df, 
                                                         save_interval, 
                                                         traj_len)
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
    # added sequentially per replica and not into the array index corresponding 
    # to the frame they were taken from. 
    

    with mda.Writer(str(Path(output_dir)/f"state_{state}.xtc"), 
                    n_atoms=sel.n_atoms) as writer:
        for i in range(coordinates.shape[0]):
            sel.positions = coordinates[i] 
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
        Path to the directory where the individual state trajectories will be 
        written.

    selection : str, optional
        An MDAnalysis selection of the atoms that should be used. Default is 
        'protein'.

    ref : str, optional
        Path to reference structure. If ref is provided, coordinates are aligned 
        to C-alphas of selection. Default is None.

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
        state_replica_frames = sort_replica_trajectories(df, save_interval, 
                                                         traj_len)
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

    with mda.Writer(str(Path(output_dir)/f"state_{state}.xtc"), 
                    n_atoms=sel.n_atoms) as writer:
        for i in range(coordinates.shape[0]):
            sel.positions = coordinates[i]  
            align.alignto(sel, ref, select=f"({selection}) and name CA")
            writer.write(sel.atoms) 




class ReplicaHandler:
    '''
    TODO : Produce a json file with hremd params that can be optionally loaded.

    A class to handle the replica exchange trajectories and state data from 
    femto HREMD simulations.
    Parameters
    ----------
    structure : str|os.PathLike
        Path to simulation topology / pdb.
    traj_dir : str|os.PathLike
        Path to directory with the femto hremd trajectories.
    hremd_data : str|os.PathLike
        Path to the femto state data output (i.e. samples.arrow file).
    save_interval : int
        The number of cycles that elapse between writing coordinates.
        femto.md.config.HREMD(trajectory_interval=save_interval)
    '''
    def __init__(self, 
                 structure:str|os.PathLike, 
                 traj_dir:str|os.PathLike, 
                 hremd_data:str|os.PathLike,
                 save_interval:int,):
        self.structure = str(structure)
        self.traj_dir = traj_dir
        self.rep_u = mda.Universe(self.structure,
                                  str(Path(self.traj_dir)/"r0.dcd"))
        self.traj_len = len(self.rep_u.trajectory)
        self.hremd_data = str(hremd_data)
        self.df = load_femto_data(hremd_data)
        self.n_states = get_num_states(self.df)
        self.save_interval = save_interval
        self.state_replica_frames = sort_replica_trajectories(self.df, 
                              self.save_interval, 
                              self.traj_len) 
        self.resources = get_resources()
        # size in bytes of all-atom trajectory
        self.traj_size = get_traj_size(self.structure,
                                       str(Path(self.traj_dir)/"r0.dcd"))

    

    def write_state_trajectory(self, 
                                        output_dir:str|os.PathLike, 
                                        selection:str='protein',
                                        ref=None,
                                        state_index=None,
                                        n_jobs=None):
    
        '''
        Separate the state trajectory for a thermodynamic state from the femto 
        HREMD replica trajectories. Each replica is assigned to a single 
        separate process.
        
        Parameters
        ----------

        output_dir : str
            Path to the directory where the individual state trajectories will 
            be written.

        selection : str, optional
            An MDAnalysis selection of the atoms that should be used. Default is
              'protein'.

        ref : str, optional
            Path to reference structure. If ref is provided, coordinates are 
            aligned to C-alphas of selection. Default is None and coordinates
            will be aligned to C-alphas of original structure.

        state_index : int
            Index of the state to write. 

        Returns
        -------
        Writes state trajectory for state_index to output_dir.
        
        '''

        if ref is not None:
            ref = mda.Universe(ref)
        else:
            ref = mda.Universe(self.structure)

        state = state_index

        out_u = mda.Universe(self.structure)
        sel = out_u.select_atoms(selection)

        coords_args_list = [(self.structure,
                            self.traj_dir,
                            self.state_replica_frames,
                            self.save_interval,
                            selection,
                            i, 
                            state)
                            for i in range(self.n_states)]
        
        if n_jobs is None:
            n_jobs = self.resources['num_cores']
        with Pool(n_jobs) as worker_pool:
            coords = worker_pool.starmap(get_state_coordinates_from_replica,
                                        coords_args_list)

        coords = [c for c in coords if c is not None]
        coordinates = np.concatenate(coords, axis=0)
        # The order of coordinates will not match the order from the original 
        # write_state_trajectories because they are added sequentially to the 
        # coordinates array as they're encountered per replica rather than into 
        # the array index corresponding to the frame they were taken from in the
        # given replica.

        with mda.Writer(str(Path(output_dir)/f"state_{state}.xtc"), 
                        n_atoms=sel.n_atoms) as writer:
            for i in range(coordinates.shape[0]):
                sel.positions = coordinates[i]  
                align.alignto(sel, ref, select=f"({selection}) and name CA")
                writer.write(sel.atoms) 

    def write_state_trajectories(self,
                                    output_dir:str|os.PathLike, 
                                    selection:str='protein',
                                    ref=None,
                                    n_jobs=None
                                    ):
    
        '''
        Write separate trajectories for each thermodynamic state from a femto 
        HREMD simulation.

        Parameters
        ----------
        output_dir : str
            Path to the directory where the individual state trajectories will 
            be written.
        selection : str, optional
            An MDAnalysis selection of the atoms that should be used. Default is
            'protein'.
        n_jobs : int, optional
            The number of parallel jobs to run. If None, will use all available 
            cores. Default is None.
        ref : str
            Path to reference structure. If ref is provided, coordinates are 
            aligned to C-alphas of selection.
        '''

        state_args_list = [(self.structure,
                            self.traj_dir,
                            self.state_replica_frames,
                            self.save_interval,
                            output_dir,
                            selection,
                            ref,
                            i) 
                            for i in range(self.n_states)] 


        if n_jobs is None:
            n_jobs = self.resources['num_cores']  
        # print(f"Attempting {n_jobs} jobs for writing state trajectories.")

        # because each job needs enough memory to hold the entire trajectory,
        # need to check available resources and set n_jobs accordingly.
        required_memory_gb = get_traj_size(self.structure,
                                       str(Path(self.traj_dir)/"r0.dcd"),
                                       selection=selection) /1e9
        # print(required_memory_gb)
        max_jobs = determine_max_jobs(required_memory_gb,
                                        self.resources['available_ram_gb'],
                                        reserve_pct=0.1)
        # if n_jobs > max_jobs:
            # # Then just use all cores for one state at a time.
            # [self.write_state_trajectory_parallel(output_dir, 
            #                                       selection, 
            #                                       ref, 
            #                                       i) 
            #                          for i in range(self.n_states)]
        n_jobs = min(n_jobs, max_jobs)
        # print(f"Using {n_jobs} jobs for writing state trajectories.")
        # run as many states as cores simultaneously

        # Apparently for Pool, the function cannot be a method of a class
        with Pool(n_jobs) as worker_pool:
            worker_pool.starmap(write_state_trajectory,
                                        state_args_list)
            



##################### Functions to get additional data from femto state data ###

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

def freq_frames(freq_file):
    '''
    TODO Move somewhere else.

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

