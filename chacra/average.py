import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations, permutations
from .utils import *
import tqdm
import os
from scipy.spatial.transform import Rotation as R
from MDAnalysis.lib.util import convert_aa_code

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

def find_identical_subunits(u: mda.Universe):
    """
    Groups segments (subunits) by identical residue sequences.

    Parameters
    ----------
    universe : mda.Universe
        Protein structure.
    """
    d = defaultdict()
    combos = combinations(range(len(u.segments)), 2)
    group = 0
    for combo in combos:
        i, j = combo
        if ''.join([convert_aa_code(k) 
                    for k in u.segments[i].residues.resnames]) == \
            ''.join([convert_aa_code(k) 
                    for k in u.segments[j].residues.resnames]):
            if len(d.values()) == 0:
                d[group] = set([i,j])
                group += 1
            else:
                found = False
                for key,val in d.items():
                    if (i in val) or (j in val):
                        d[key].add(i)
                        d[key].add(j)
                        found = True
                if found == False:
                    d[group] = set([i,j])
                    group += 1
                
    return {key:list(val) for key, val in d.items()}

def same_list_membership(d: dict[int, list[str]], a: int, b: int) -> bool:
    '''
    Check if both `a` and `b` appear in the same list of the dictionary.
    
    Returns True if they are in the same list, False if in different lists
    or only one is found.
    '''
    for group in d.values():
        if a in group and b in group:
            return True
    return False

def align_principal_axes_to_global_frame(u:mda.Universe) -> np.ndarray:
    '''
    Reorients the Universe in place to align with the global axes.
    '''
    coords = u.atoms.positions - u.atoms.center_of_mass()
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    local_frame = eigvecs  # columns are basis vectors
    global_frame = np.eye(3)  # identity: x, y, z axes

    R = global_frame @ local_frame.T

    universe_com = u.atoms.center_of_mass()
    shifted = u.atoms.positions - universe_com
    rotated = shifted @ R.T

    u.atoms.positions = rotated 

def get_long_axis(subunit: mda.AtomGroup) -> np.ndarray:
    coords = subunit.positions - subunit.center_of_mass()
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Longest axis corresponds to largest eigenvalue
    long_axis = eigvecs[:, np.argmax(eigvals)]
    return long_axis / np.linalg.norm(long_axis)

def align_mobile_to_ref(mobile: mda.AtomGroup, ref: mda.AtomGroup):
    '''
    Aligns the long axis of mobile atom group to ref atom group's in place.
    '''
    # Compute COMs and long axes
    ref_com = ref.center_of_mass()
    mob_com = mobile.center_of_mass()

    ref_axis = get_long_axis(ref)
    mob_axis = get_long_axis(mobile)

    # Compute rotation matrix to align mobile axis to reference axis
    v = np.cross(mob_axis, ref_axis)
    c = np.dot(mob_axis, ref_axis)
    s = np.linalg.norm(v)

    if s == 0:  # Vectors already aligned or anti-aligned
        if c > 0:
            R = np.eye(3)
        else:
            # 180 degree rotation: find orthogonal axis to rotate around
            ortho = np.eye(3)[np.argmin(np.abs(mob_axis))]  # any axis not aligned
            v = np.cross(mob_axis, ortho)
            v /= np.linalg.norm(v)
            K = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            R = np.eye(3) + 2 * K @ K
    else:
        v /= s  # normalize rotation axis
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))

    # Apply transformation: move COM to origin, rotate, then translate to ref COM
    shifted = mobile.positions - mob_com
    rotated = shifted @ R.T
    aligned = rotated + ref_com
    mobile.positions = aligned

def get_rotation_matrix(chaina:mda.AtomGroup, 
                        chainb:mda.AtomGroup,
                        ) -> np.ndarray:
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
    matrix, rmsd = align.rotation_matrix(chaina.positions, 
                                         chainb.positions,
                                         )

    return matrix

def get_all_rotations(u:mda.Universe,
                      identical_subunits:dict,
                      ) -> dict[int, dict[int, np.ndarray]]:
    
    '''
    Get all of the rotations for subunit i to subunits j to N for all subunits.

    u : mda.Universe
        Should contain only symmetric protein complexes with identical numbers
        of subunits for all subunit types.
    
    identical_subunits : dict
        The dictionary of segment id lists where each list only contains
        the ids of subunits that are of identical types. 
        i.e. find_identical_subunits(u) 

    Returns
    -------
    dictionary
        The dictionary of rotation matrices with keys corresponding to seg ids
        and values of dictionaries with the segid of that the subunit was 
        rotated into and the corresponding rotation matrix.
        e.g. {0:{1:np.ndarray}} is the np.ndarray rotation matrix
        of segment 0 to segment 1
    '''
    segids = [i for i in range(len(u.segments))]#
    
    align_principal_axes_to_global_frame(u)
   
    seg_combos = [combo for combo in permutations(segids,2)]
    seg_chain = seg_to_chain(u)
    rotations = {seg_chain[segid1]: 
                 {seg_chain[segid2]:0 for segid2 in segids
                  if segid2 != segid1}
                  for segid1 in segids}
    
    for combo in seg_combos:
        # rotating sega onto segb
        sega, segb = combo
        # u2 is rotated. u stays aligned to global frame
        u2 = u.copy()
        # if they're identical subunits
        if same_list_membership(identical_subunits,
                                sega, 
                                segb):
            # just use first instance of the chain type as ref for now
            ref = [val for val in identical_subunits.values()
                   if sega in val][0][0]
            # position sega on the reference segment
            
            align.alignto(u2.segments[sega].atoms, 
                      u.segments[ref].atoms,
                      select="name CA",)

            ca_sela = u2.select_atoms(f"chainID {seg_chain[sega]} and name CA")
            ca_selb = u2.select_atoms(f"chainID {seg_chain[segb]} and name CA")
            rotations[seg_chain[sega]][seg_chain[segb]] = get_rotation_matrix(
                                                        ca_sela,
                                                        ca_selb,
                                                        )
        else: # they're non-identical subunits
            # so if we're interested in the relationship of type a to type b
            # we need to position a copy of type b on top of type a's reference
            # and use this copy to collect the rotation of type a onto type b
            # reference subunit of type a
            refa = [val for val in identical_subunits.values()
                   if sega in val][0][0]
            align.alignto(u2.segments[sega].atoms, 
                      u.segments[refa].atoms,
                      select="name CA",)
            # copy of type b always use the ref type b
            refb = [val for val in identical_subunits.values()
                   if segb in val][0][0]
            mobile = mda.Merge(u2.segments[refb].atoms)
            # move mobile's com to hetero ref com and align their long axes.
            align_mobile_to_ref(mobile.atoms, u.segments[refa].atoms)

            ca_sela = mobile.select_atoms(f"name CA")
            ca_selb = u2.select_atoms(f"chainID {seg_chain[segb]} and name CA")
            rotations[seg_chain[sega]][seg_chain[segb]] = get_rotation_matrix(
                                                        ca_sela,
                                                        ca_selb,
                                                        )
    return rotations

def get_equivalent_interactions(array_dict:dict[str,dict[str,np.ndarray]],
                                identical_subunits:dict[int,list[int]],
                                chain_seg_dict:dict[int,str],
                                representative_chains:list=None) -> dict:
    '''
    Map a pair of subunits to other pairs of subunits with the same
    spatial relationship. 

    Parameters
    ----------
    array dict : dict
        Keys are chainid1 with dictionary values. The sub dictionaries 
        give chainid2 and its rotation matrix relationship to chainid1. 

    identical_subunits : dict
        Keys are integers and values are lists of segment ids that are 
        identical.

    chain_seg_dict : dict
        Keys are chain id strings and values are the corresponding segment id 
        integer.

    representative_chains : list
        If a list of chainIDs is provided, the returned dictionary will only
        include non-redundant key, value pairs where the key involves a 
        chain in the representative_chains list.

    Returns
    -------
    Dictionary
    Mapping of the relationships of chains x to y (values) that are equivalent 
    to chain i to j (keys)
    '''
    
    d = defaultdict(set)
    # take a subunit (ch1) and it's rotations with all others in matrix_dict1
    for ch1, matrix_dict1 in array_dict.items():
        # take each subunit in the matrix_dict and ch1's rotation onto cha
        for cha, matrixa in matrix_dict1.items():
            # take all other subunits' rotations onto all others
            for ch2, matrix_dict2 in array_dict.items():
                # if ch1 onto X, don't consider ch1 onto anything else
                if ch2 == ch1:
                    continue
                # ensure that we're only looking at mappings of other subunits
                # that are identical to ch1 type
                elif not same_list_membership(identical_subunits,
                                              chain_seg_dict[ch1],
                                              chain_seg_dict[ch2]):
                    continue
                
                chb = np.argmin([
                (np.linalg.norm(matrixa - matrixb, ord='fro')) 
                for matrixb in matrix_dict2.values()])

                # get the actual subunit name string by indexing it from the argmin
                chb = list(matrix_dict2.keys())[chb]
                # ensure that what cha is being mapped onto is coming from the same
                # type of subunits as what chb is mapped onto
                if not same_list_membership(identical_subunits,
                                            chain_seg_dict[cha],
                                            chain_seg_dict[chb]):
                    continue

                d[(ch1,cha)].add((ch2,chb))

    seg_chain_dict = {seg:chain for chain, seg in chain_seg_dict.items()}
    # make the entries for self-similars
    for chain_list in identical_subunits.values():
        d |= {(seg_chain_dict[seg],seg_chain_dict[seg]): 
              set([(seg_chain_dict[seg2],seg_chain_dict[seg2])
                   for seg2 in chain_list if seg2 != seg]) 
                   for seg in chain_list}
    
    # last, add the key to it's own set
    for pair in d:
        d[pair].add(pair)
    # convert to lists
    d = {key: list(val) 
            for key, val in d.items()}
    if representative_chains is not None:
        # TODO add check to get rid of redundant lists
        # e.g. if a-b is in list 1 and list 2 drop key:list2
        # but you have to consider representative_chains and sorted elements
        # because in a trimer AB and AC will be two keys but map to the same
        # pairs (when sorted)
        eq_int = {}
        for ch_pair, pair_list in d.items():
            if (ch_pair[0] in representative_chains):
                eq_int[ch_pair] = pair_list
        d = eq_int
    return d

def validate_group_memberships(L:list, D:dict) -> bool:
    '''
    Ensure that each element of L occurs only once in
    one list value of D for all lists in D

    L : list
        list of chain IDs that will be used as a representative chains
        in a heteromultimer
    D : dictionary
        values of D are lists of identical subunits
    
    Returns
    -------
    bool
    '''
    # Map: group ID -> count of L-elements in that group
    group_counts = {k: 0 for k in D}

    # Create a reverse index: element -> group ID
    element_to_group = {}
    for k, values in D.items():
        for v in values:
            element_to_group[v] = k

    for item in L:
        if item not in element_to_group:
            return False  # Item not found in any group
        group_counts[element_to_group[item]] += 1

    # Every group must be represented exactly once
    return all(count == 1 for count in group_counts.values())


def get_chain_group(chain, identical_subunits):
    '''
    Return the identical subunit group that the subunit is in
    '''
    for group in identical_subunits:
        if chain in identical_subunits[group]:
            return group
        

# def make_equivalent_contact_regex(resids):
#     '''
#     resids : the parse_id dictionary containing the contact data
#     #TODO can remove any potential ambiguity by adding the list of correct chain 
#     group chains 
#     '''
#     regex1 = rf"[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)-[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)"
#     regex2 = rf"[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)-[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)"
#     return rf"{regex1}|{regex2}"

def alphabetize(contact_name:str) -> str:
    a, b = contact_name.split("-")
    if a<b:
        return contact_name
    else:
        return f"{b}-{a}"
    
def map_chains(chain_pair:tuple, pairs_list:list) -> dict:
    '''
    Maps alphabetically ordered chains to their relationship order with other
    equivalently interacting chains. Used to generate the correct contact names.

    chain_pair : tuple
        The alphabetically ordered tuple of chain IDs

    pairs_list : list
        The list of chain pair tuples from the equivalent_interactions
        that contain the relative orientation information rather than being
        alphabetically ordered.

    Returns
    -------
    Dictionary
    The mapping of the input chain pairs to their ordered relationship

    '''
    for pair in pairs_list:
        if chain_pair[0] in pair and chain_pair[1] in pair:
            if chain_pair[0] == pair[0]:
                chain_map = {0:'a',1:'b'}
            else:
                chain_map = {0:'b',1:'a'}
    return chain_map

def make_equivalent_contact_names(resinfo:dict,
                                  equivalent_interactions:dict) -> list:
    '''
    Uses the orientation aware equivalent_interactions that is already reduced
    to the keys involving the representative_chains to make the names for the 
    contacts that are equivalent to the resinfo contact.

    resinfo : dict
        The dictioinary of contact info from parse_id

    equivalent_interactions : dict
    
    Returns
    -------
    List of equivalent contacts to average.
    '''
    in_pair = (resinfo['chaina'],resinfo['chainb'])
    for key, pair_list in equivalent_interactions.items():
        if in_pair in [tuple(sorted(pair)) for pair in pair_list]:
            chain_pairs = equivalent_interactions[key]
    # determine which tuple element corresponds to chaina and chainb
    chain_map = map_chains(in_pair, chain_pairs)
   
    contacts = []
    for pair in chain_pairs:
        contacts.append(f"{pair[0]}:{resinfo[f'resn{chain_map[0]}']}:{resinfo[f'resid{chain_map[0]}']}-"\
                     f"{pair[1]}:{resinfo[f'resn{chain_map[1]}']}:{resinfo[f'resid{chain_map[1]}']}")

    contacts = [alphabetize(contact) for contact in contacts]
    return list(set(contacts))

def get_representative_name(resinfo, 
                            equivalent_interactions):
    '''
    provide the chains from the contact and get the chain names to use for 
    making a generalized/averaged contact name
    '''

    chaina, chainb = resinfo['chaina'],resinfo['chainb']
    for pair, eq_pairs in equivalent_interactions.items():
        if (chaina,chainb) in [tuple(sorted(pair2)) for pair2 in eq_pairs]:
            representative_pair = tuple(sorted(pair))
            chain_list = equivalent_interactions[pair]
            break
    chain_map = map_chains((chaina,chainb),chain_list)
    name = f"{pair[0]}:{resinfo[f'resn{chain_map[0]}']}:{resinfo[f'resid{chain_map[0]}']}-"\
            f"{pair[1]}:{resinfo[f'resn{chain_map[1]}']}:{resinfo[f'resid{chain_map[1]}']}"                     
    
    
    return alphabetize(name)

def get_pair_distance(sel1, sel2, u):
    '''
    Return the distance between two selections
    '''
    a = u.select_atoms(sel1).positions[0]
    b = u.select_atoms(sel2).positions[0]
    return np.linalg.norm(a-b)

#################### Main Function ##########################

def average_multimer(structure: str|os.PathLike, 
                     df: pd.DataFrame, 
                     representative_chains: list=None,
                     return_stdev: bool=False, 
                     ) -> pd.DataFrame:
    '''
    Takes a pdb structure, contact df, and list of chain(s) to use for 
    representing the averaged contacts. Finds all of the equivalent 
    interactions and returns a dataframe of the averaged values. The subunits in
    the structure should be symmetrically arranged and if it's a heteromultimer,
    there should be an identical number of each type of subunit.

    Parameters
    ----------
    structure : str
        Path to pdb file that contact frequencies/ simulations are based on.

    df : pd.DataFrame
        The contact frequency dataframe to average.
    
    representative_chains : list
        List of pdb chain ids to use for depicting the averaged contacts on.
        If structure is a homomultimer, you do not need to specify this and the
        chain corresponding to the first segment in the universe is used.
        If it's a heteromultimer then specify the closest heterosubunit to 'A' 
        e.g. ['A','F']
    
    return_stdev : bool
        If True, returns averaged_contacts and standard_deviation of contact 
        frequencies for identical interactions. 
        Usage : avg_df, stdev = average_multimer(args)
   
    Returns
    -------
    pd.DataFrame 
    '''
    structure = str(structure)
    u = mda.Universe(structure)
    u = mda.Merge(u.select_atoms('protein'))
    ###### Only averaging contacts for protein
    
    # int key to list of int segids
    identical_subunits = find_identical_subunits(u)
    chain_seg = chain_to_seg(u) # chain str to segid dict
    
    # Checks and representative chain assignments if None
    if has_only_identical_subunits(u):
        if representative_chains is None:
            representative_chains = [u.segments.segids[0]]
    else: # heteromultimer
        if len(identical_subunits) > 1:
            test_len = len(list(identical_subunits.values())[0])
            if not np.alltrue([len(val)==test_len 
                                for val in identical_subunits.values()]):
                print("Can't average a heteromultimer with an unequal number " \
                "of each type of subunit.")
                return
            elif len(representative_chains) != len(identical_subunits):
                print("You need to specify 1 chain id for each type of subunit "\
                "in the representative_chains argument. They should be in " \
                "contact in the structure.")
                return
            elif not validate_group_memberships(representative_chains, 
                                           identical_subunits):
                print("The representative chains don't come from unique sets "\
                      "of subunits in the identical_subunits dictionary.")
                return
            if representative_chains is None: # need to base this off of nearest subunits
                representative_chains = [val[0] for val 
                                     in identical_subunits.values()]
                   
    # ensure correct averaging       
    denominator = len(identical_subunits[0])
    # for identifying equivalent spatial relationships between subunits
    rotations = get_all_rotations(u, identical_subunits)
    # non alphabetically sorted dictionary where key (B,A) has list of
    # chain pairs that are equivalent to A's relationship to B
    # this let's you directly assign which contact names to search for.
    # reduce equivalent_interactions to only keys with representative_chains
    equivalent_interactions = get_equivalent_interactions(rotations,
                                                        identical_subunits,
                                                        chain_seg,
                                                        representative_chains)
    # removing the columns in place, so using a copy of the original data
    df_copy = df.copy()
    # hold the averaged data
    averaged_data = {}
     # collect this for error bars on averaged contact vs temp plots
    standard_deviation = {}
    # TODO keep track of what's averaged
    ############## Main Loop Begins here ###################
    print('Collecting equivalent contacts and averaging.\n')
    total_count = len(df.columns)
    with tqdm.tqdm(total=total_count) as progress:
        # as equivalent contacts are averaged, the columns are removed
        while len(df_copy.columns) > 0:
            # take the first of the remaining column names as the template
            resinfo = parse_id(df_copy.columns[0])
            # Create the name that the averaged value will be associated with
            averaged_name = get_representative_name(resinfo, 
                                                    equivalent_interactions)
            # make the names for all the equivalent contacts to average
            to_average = make_equivalent_contact_names(resinfo, 
                                                       equivalent_interactions)
            # Get only the contact names that exist in the dataframe
            to_average = np.intersect1d(to_average, df_copy.columns.values)

            # keep track of anything that shouldn't be happening
            if len(to_average) > denominator:
                print(f"averaging {len(to_average)} contacts "\
                      f"when there should be at most {denominator}. "\
                      f"contacts are : {to_average}")
                averaged_data[averaged_name] = df_copy[to_average].sum(
                                                        axis=1)/len(to_average)
            else:
                averaged_data[averaged_name] = df_copy[to_average].sum(
                                                            axis=1)/denominator

            standard_deviation[averaged_name] = df_copy[to_average].std(axis=1)
            # get rid of the contacts that were just averaged and reduce the 
            # number of remaining contacts to check in the dataframe
            df_copy.drop(to_average, axis=1, inplace=True)
            # update the progress bar
            columns_removed = len(to_average)
            progress.update(columns_removed)
    if return_stdev == True:
        return pd.DataFrame(averaged_data), pd.DataFrame(standard_deviation)
    else:
        return pd.DataFrame(averaged_data)
    

def everything_from_averaged(averaged_contacts:pd.DataFrame, 
                            u:mda.Universe, 
                            representative_chains:list[str],
                            as_map:bool=False):
    '''
    Take the averaged contacts and regenerate the entire protein's contacts 
    using this data. Useful for visualizing the flow of chacras across the whole 
    protein and doing network analysis on more statistically robust data.

    averaged_contacts : pd.DataFrame
        The averaged contact data.

    original_contacts : pd.DataFrame
        The original contact dataframe.

    u : MDA.Universe 
        The same universe used for averaging contacts

    representative_chains : list
        The list of chain ids used when generating the averaged contact names.

    as_map : bool
        Returns a dictionary mapping each averaged contact name to a list of the
        corresponding replicated contact names


    Returns
    -------
    pd.DataFrame of the averaged contact values applied back to all chains.
    '''
    print("Collecting some information. One moment.")
    protein = u.select_atoms('protein')
    u = mda.Merge(protein)
    
    identical_subunits = find_identical_subunits(u)
    chain_seg = chain_to_seg(u)
    rotations = get_all_rotations(u,identical_subunits)
    equivalent_interactions = get_equivalent_interactions(rotations,
                                                           identical_subunits,
                                                           chain_seg,
                                                          representative_chains)


    replicated_contacts = {}
    mapped_contacts = {contact:[] for contact in averaged_contacts.columns}
    
    for averaged_contact in tqdm.tqdm(averaged_contacts.columns):
        resinfo = parse_id(averaged_contact)
        remade_contacts = make_equivalent_contact_names(resinfo,
                                                 equivalent_interactions)
        for remade_contact in remade_contacts:
            replicated_contacts[remade_contact] = \
                averaged_contacts[averaged_contact]
    
        mapped_contacts[averaged_contact] = remade_contacts
                    
    if as_map:
         return mapped_contacts
    else:
        return pd.DataFrame.from_dict(replicated_contacts)
