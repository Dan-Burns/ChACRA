import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations, permutations
#from .ContactFrequencies import *
from .utils import *
import tqdm
import os
from scipy.spatial.transform import Rotation as R
import re
from Bio.SVDSuperimposer import SVDSuperimposer

###### Checking #########
def has_only_identical_subunits(u: mda.Universe) -> bool:
    """
    Check if all subunits in the universe are identical.
    
    Parameters
    ----------
    u : mda.Universe
        The MDAnalysis universe to check.
    
    Returns
    -------
    bool
        True if all subunits are identical, False otherwise.
    """
    seqs = [tuple(seg.residues.resnames) for seg in u.segments]
    if not all(s == seqs[0] for s in seqs):
        raise ValueError("Not all subunits are identical. This function only"\
                         " supports homomultimers.")
    else:
        return True
    
###### Preps and helpers ################
def seg_to_chain(u:mda.Universe) -> dict[int,str]:
    '''
    Returns the seg id to chain id mapping.

    u : mda.Universe

    Returns
    -------
    dict
    {int(segid): str(chainid)}
    '''
    return {i: list(set(seg.atoms.chainIDs))[0] 
            for i,seg in enumerate(u.segments)}

def chain_to_seg(u:mda.Universe) -> dict[str,int]:
    '''
    Returns the seg id to chain id mapping.

    u : mda.Universe

    Returns
    -------
    dict
    {int(segid): str(chainid)}
    '''
    return {list(set(seg.atoms.chainIDs))[0]: i
            for i,seg in enumerate(u.segments)}



######### alignments for rotations ##########

def align_principal_axes_to_global_frame(u:mda.Universe) -> np.ndarray:
    coords = u.atoms.positions - u.atoms.center_of_mass()
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Orthonormal frame defined by principal axes
    local_frame = eigvecs  # columns are basis vectors

    # Global frame basis
    global_frame = np.eye(3)  # identity: x, y, z axes

    # Rotation matrix that maps local_frame to global_frame:
    R = global_frame @ local_frame.T

    universe_com = u.atoms.center_of_mass()
    shifted = u.atoms.positions - universe_com

    rotated = shifted @ R.T

    u.atoms.positions = rotated 

    



####### rotations ################
def get_rotation_matrix(chaina:mda.AtomGroup, 
                        chainb:mda.AtomGroup,
                        svd:bool=False) -> np.ndarray:
    '''
    Get the rotation matrix that aligns chaina with chainb

    u : mda.Universe

    chaina : mda.AtomGroup
        chaina and chainb should be identical subunits / chains.

    chainb : mda.AtomGroup

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    '''
    # a_ca_mask = np.where(chaina.names == "CA")
    # b_ca_mask = np.where(chainb.names == "CA")

    if svd is True:
        sup = SVDSuperimposer()
        sup.set(chainb.positions, chaina.positions) # second element is rotated and translated onto the 1st
        sup.run()
        matrix, tran = sup.get_rotran()
       
    else:
        matrix, rmsd = align.rotation_matrix(chaina.positions, 
                                         chainb.positions)

    return matrix

def get_all_rotations(u:mda.Universe,
                      reference_segment:int=0
                      ) -> dict[int, dict[int, np.ndarray]]:
    
    '''
    Get all of the rotations for subunit i to subunits j to N for all subunits
    in a homomultimeric universe.

    u : mda.Universe

    Returns
    -------
    dictionary
        The dictionary of rotation matrices with keys corresponding to seg ids
        and values of dictionaries with the segid of that the subunit was 
        rotated into and the corresponding rotation matrix.
        e.g. {0:{1:np.ndarray}} is the np.ndarray rotation matrix
        of segment 0 to segment 1
    '''

    
    if has_only_identical_subunits(u):
        segids = [i for i in range(len(u.segments))]

    align_principal_axes_to_global_frame(u)
    seg_combos = [combo for combo in permutations(segids,2)]
    seg_chain = seg_to_chain(u)
    rotations = {seg_chain[segid1]: 
                 {seg_chain[segid2]:0 for segid2 in segids
                  if segid2 != segid1}
                  for segid1 in segids}
    
    for combo in seg_combos:
        sega, segb = combo
        u2 = u.copy()
        align.alignto(u2.segments[sega].atoms, 
                      u.segments[reference_segment].atoms,
                      match_atoms=False)
        rotations[seg_chain[sega]][seg_chain[segb]] = get_rotation_matrix(
                                                    u2.segments[sega].atoms,
                                                    u2.segments[segb].atoms)
    return rotations

# def create_com_surrounding_points_universe(u: mda.Universe,
#                                            distance: float = 10.0
#                                            ) -> mda.Universe:
#     """
#     Create a new Universe where each subunit is represented by 6 points
#     at ±distance from its center of mass along the global x, y, and z axes.

#     Parameters
#     ----------
#     u : mda.Universe
#         The original universe containing multiple subunits.
#     distance : float
#         Distance from the COM to place the dummy atoms.

#     Returns
#     -------
#     new_u : mda.Universe
#         A new universe containing only the 6-point representations for each chain.
#     """
#     point_atoms = []
#     for i, segment in enumerate(u.segments):
#         com = segment.atoms.center_of_mass()

#         # Generate 6 positions ±x, ±y, ±z from the COM
#         offsets = np.eye(3) * distance  # [[x,0,0], [0,y,0], [0,0,z]]
#         directions = np.concatenate([offsets, -offsets], axis=0)  # 6 total

#         coords = com + directions  # shape (6, 3)

#         # Create fake AtomGroup using MDAnalysis.Merge
#         # We create 6 atoms per segment, all with unique resid and name
#         atoms = mda.Universe.empty(n_atoms=6,
#                                    n_residues=6,
#                                    atom_resindex=np.arange(6),
                                   
#                                    n_segments=1,
#                                    #names=["P"]*6,
#                                    #residue_names=["DUM"]*6,
#                                 #    segment_ids=[segment.segid]),
#                                 trajectory=True
#         )
        
#         atoms.add_TopologyAttr('segid', [i])
#         atoms.add_TopologyAttr('mass', np.ones(6))
#         atoms.atoms.positions = coords
#         point_atoms.append(atoms)

#     # Merge all per-segment 6-atom groups into a single universe
#     new_u = mda.Merge(*[ag.atoms for ag in point_atoms],)
#     return new_u

# def get_all_rotations_hetero(u:mda.Universe
#                       ) -> dict[int, dict[int, np.ndarray]]:
    
#     '''
#     Doesn't Work.

#     Get all of the rotations for subunit i to subunits j to N for all subunits
#     in a symmetric heteromultimeric assembly.

#     u : mda.Universe

#     Returns
#     -------
#     dictionary
#         The dictionary of rotation matrices with keys corresponding to seg ids
#         and values of dictionaries with the segid of that the subunit was 
#         rotated into and the corresponding rotation matrix.
#         e.g. {0:{1:np.ndarray}} is the np.ndarray rotation matrix
#         of segment 0 to segment 1
#     '''

    
   
#     segids = [i for i in range(len(u.segments))]

#     seg_combos = [combo for combo in permutations(segids,2)]
#     seg_chain = seg_to_chain(u)
#     rotations = {seg_chain[segid1]: 
#                  {seg_chain[segid2]:0 for segid2 in segids
#                   if segid2 != segid1}
#                   for segid1 in segids}
    
#     for combo in seg_combos:
#         sega, segb = combo
#         u2 = u.copy()
#         align_universe_by_subunit(u2, u2.segments[sega].atoms)
#         com_u = create_com_surrounding_points_universe(u)
        
#         rotations[seg_chain[sega]][seg_chain[segb]] = get_rotation_matrix(
#                                                     com_u.segments[sega].atoms,
#                                                     com_u.segments[segb].atoms)
#     return rotations

# def get_all_com_vectors(u: mda.Universe,) -> dict[str,dict[str,np.ndarray]]:
#     """
#     Compute vectors from the center of mass of a reference subunit to all others.

#     Parameters
#     ----------
#     u : mda.Universe
    

#     Returns
#     -------
#     dict
#         Directional vector of each chains center of mass to every other chains.
#     """
#     u = u.copy()
#     segids = [i for i in range(len(u.segments))]

#     seg_combos = [combo for combo in permutations(segids,2)]
#     seg_chain = seg_to_chain(u)
    
#     vectors = {seg_chain[segid1]: 
#                  {seg_chain[segid2]:0 for segid2 in segids
#                   if segid2 != segid1}
#                   for segid1 in segids}
#     for combo in seg_combos:
#         sega, segb = combo
        
#         u2 = u.copy()
#         align_universe_by_subunit(u2, u2.segments[sega].atoms) # can expose axis
#         ref_com = u2.segments[sega].atoms.center_of_mass()
#         vectors[seg_chain[sega]][seg_chain[segb]] = u2.segments[
#                                         segb].atoms.center_of_mass() - ref_com
#     return vectors



def get_equivalent_pairs(array_dict:dict[str,dict[str,np.ndarray]]) -> dict:
    '''
    Map a pair of subunits to other pairs of subunits with the same
    spatial relationship. 

    Parameters
    ----------
    array dict : dict
        Keys are chainid1 with dictionary values. The sub dictionaries 
        give chainid2 and its vector relationship to chainid1. 
    '''
    d = defaultdict(set)
    for ch1, matrix_dict1 in array_dict.items():
        # vector relating chain1 to chaina
        for cha, matrixa in matrix_dict1.items():
            if len(matrixa.shape)>1:
                ord = 'fro'
            else:
                ord = None
            for ch2, matrix_dict2 in array_dict.items():
                if ch2 == ch1:
                    continue
                # find the relationship between seg2 and segb that is most 
                # similar to the relationship between chain1 and chaina
                # if using directional vectors, can apply to heterosubunits
                # if using rotations, the array dict must correspond to only
                # homomeric assemblies

                # if not isinstance(matrixb, int) 
                # else np.inf 
                chb = np.argmin([
                (np.linalg.norm(matrixa - matrixb, ord=ord)) 
                for matrixb in matrix_dict2.values()])

                # chb = np.argmin([
                # (matrixa - matrixb).sum().sum()
                # for matrixb in matrix_dict2.values()])

                # chb = np.argmin([
                # np.allclose(matrixa,matrixb, rtol=1e-2) 
                # for matrixb in matrix_dict2.values()])

                chb = list(matrix_dict2.keys())[chb]

                d[(ch1,cha)].add((ch2,chb))
    return d
