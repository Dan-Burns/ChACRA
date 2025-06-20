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

# def get_all_rotations(u:mda.Universe,
#                       reference_segment:int=0,
#                       identical_subunits:dict=None,
#                       ) -> dict[int, dict[int, np.ndarray]]:
    
#     '''
#     Get all of the rotations for subunit i to subunits j to N for all subunits
#     in a homomultimeric universe.

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
#     if has_only_identical_subunits(u):
#         segids = [i for i in range(len(u.segments))]#

#     align_principal_axes_to_global_frame(u)

#     seg_combos = [combo for combo in permutations(segids,2)]
#     seg_chain = seg_to_chain(u)
#     rotations = {seg_chain[segid1]: 
#                  {seg_chain[segid2]:0 for segid2 in segids
#                   if segid2 != segid1}
#                   for segid1 in segids}
    
#     for combo in seg_combos:
#         # rotating sega onto segb
#         sega, segb = combo
#         # u2 is rotated, u stays aligned with global frame
#         u2 = u.copy()
#         # every subunit is aligned to reference subunit before saving rotation
#         align.alignto(u2.segments[sega].atoms, 
#                       u.segments[reference_segment].atoms,
#                       match_atoms=False)
#         # then rotate sega onto segb given the same orientation as ref with 
#         # everything else
#         rotations[seg_chain[sega]][seg_chain[segb]] = get_rotation_matrix(
#                                                     u2.segments[sega].atoms,
#                                                     u2.segments[segb].atoms)
#     return rotations

def get_all_rotations(u:mda.Universe,
                      identical_subunits:dict=None,
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
            ref = [val for val in identical_subunits.values()
                   if sega in val][0][0]
            align.alignto(u2.segments[sega].atoms, 
                      u.segments[ref].atoms,
                      select="name CA",)
            # copy of type b
            mobile = mda.Merge(u2.segments[segb].atoms)
            # move mobile's com to hetero ref com and align their long axes.
            align_mobile_to_ref(mobile.atoms, u2.segments[ref].atoms)

            ca_sela = mobile.select_atoms(f"name CA")
            ca_selb = u2.select_atoms(f"chainID {seg_chain[segb]} and name CA")
            rotations[seg_chain[sega]][seg_chain[segb]] = get_rotation_matrix(
                                                        ca_sela,
                                                        ca_selb,
                                                        )
            print(f"end {combo}")
    return rotations

def get_equivalent_interactions(array_dict:dict[str,dict[str,np.ndarray]],
                                identical_subunits:dict[int,list[int]],
                                chain_seg_dict:dict[int,str],
                                sorted:bool=False) -> dict:
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

    sorted : bool
        If True, all of the chain ID tuples will be alphabetically sorted. This
        eliminates the orientation relationship information. 
        Default is False.

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
    if sorted is True:
        d = {tuple(sorted(key)): [tuple(sorted(v)) for v in val] 
             for key, val in d.items()}
    else:
        d = {key: list(val) 
             for key, val in d.items()}
    return d

def get_long_axis(subunit: mda.AtomGroup) -> np.ndarray:
    coords = subunit.positions - subunit.center_of_mass()
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Longest axis corresponds to largest eigenvalue
    long_axis = eigvecs[:, np.argmax(eigvals)]
    return long_axis / np.linalg.norm(long_axis)

def align_mobile_to_ref(mobile: mda.AtomGroup, ref: mda.AtomGroup) -> None:
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


def validate_group_memberships(L, D):
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
        

def make_equivalent_contact_regex(resids):
    '''
    resids : the parse_id dictionary containing the contact data
    #TODO can remove any potential ambiguity by adding the list of correct chain 
    group chains 
    '''
    regex1 = rf"[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)-[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)"
    regex2 = rf"[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)-[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)"
    return rf"{regex1}|{regex2}"

def get_representative_pair_name(chaina, chainb, identical_subunits, 
                                 representative_chains, equivalent_interactions):
    '''
    provide the chains from the contact and get the chain names to use for 
    making a generalized/averaged contact name
    '''
    representative_pair = None
    if chaina == chainb:
        for key, identical_subunit_list in identical_subunits.items():
            if chaina in identical_subunit_list:
                for representative_chain in representative_chains:
                        if representative_chain in identical_subunit_list:
                            representative_pair = (representative_chain, 
                                                    representative_chain)
                                
    # determine which equivalent_interaction set it came from 
    else:
        paira = (chaina,chainb)
        
        for representative_pair_name, equivalent_interaction_list in \
        equivalent_interactions.items():
            for pair in equivalent_interaction_list:
                # The current version maintains the relationship
                # rather than being alphabetical like the incoming contact
                # so have to sort pair
                if paira == pair: 
                        representative_pair = representative_pair_name
                        
                        break
    if representative_pair is None:
        raise ValueError(f"Could not determine representative_pair for ({chaina}, {chainb})")
    
    return representative_pair

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
                     return_stdev:bool=False, 
                     
                     ) -> pd.DataFrame:
    '''
    
    Takes a contact df, pdb structure, and a selection of chains to use for 
    representing the averaged contacts and finds all of the equivalent 
    interactions and returns a dataframe of the averaged values.  These values 
    are more robust and easier to visualize. The subunits in the structure shoul
    be identical and symmetrically arranged. Hetero multimers with identical 
    numbers of each type of subunit should be processed correctly as well.

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

    check_folder : str
        Name of folder to write verification files to. These should be used with
        pymol to confirm that the averaging is being done between the correct 
        pairs of subunits.
   
    Returns
    -------
    pd.DataFrame containing the values for averaged among the identical contacts.


    
    '''
    structure = str(structure)
    u = mda.Universe(structure)
    protein = u.select_atoms('protein')
    ###### Only averaging contacts for protein
    u = mda.Merge(protein)

    identical_subunits = find_identical_subunits(u)
    seg_chain = seg_to_chain(u)
    chain_seg = chain_to_seg(u)
    denominator = len(identical_subunits[0])

    

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
                print("You need to specify 1 chain id for each type of subunit " \
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
                   
            
    
    rotations = get_all_rotations(u, identical_subunits)
    equivalent_interactions = get_equivalent_interactions(rotations,
                                                        identical_subunits,
                                                        chain_seg)
    # removing the columns in placed, so using a copy of the original data
    df_copy = df.copy()
    # hold the averaged data
    averaged_data = {}
     # collect this for error bars on averaged contact vs temp plots
    standard_deviation = {}

    ############## Main Loop Begins here ###################
    print('Collecting equivalent contacts and averaging.\n')
    total_count = len(df.columns)
    with tqdm.tqdm(total=total_count) as progress:
        # as equivalent contacts are averaged, the original columns are removed
        while len(df_copy.columns) > 0:
            # take the first of the remaining column names as the template
            resids = parse_id(df_copy.columns[0])

            # create the regular expression that will be used to match
            # equivalent contacts in the rest of the dataframe
            regex = make_equivalent_contact_regex(resids)
            # find all of the other (equivalent) contacts that involve these 
            # residue names and numbers
            to_average = list(df_copy.filter(regex=regex, axis=1).columns)
            # chaina and chainb are picked up from this iterations initial contact 
            chaina, chainb = resids['chaina'], resids['chainb']
            if chaina == chainb:
                identical_pair = True # confusing - identical_subunits dictionary refers to different IDs but this indicates same ID
            else:
                identical_pair = False

            # then this chain pair is mapped back to the
            # the representative chains
            representative_pair = get_representative_pair_name(
                                                        chaina, 
                                                        chainb, 
                                                        identical_subunits, 
                                                        representative_chains, 
                                                        equivalent_interactions
                                                            )
            ################ Determine Average Contact Name and
            ################## Filter out things that shouldn't be in to_average 
                # The point of all this is that the contact that
                # involves the representative chain might not have occurred
                # during the simulations on that subunit, but did on others.
                # If this is the case, you have to identify which residue goes
                # on which chain of the contact involving the representative pair
                # there is probably a more efficient way....
                
            ############### Happening in same subunit ##########################
            if identical_pair:
                # This is the easiest (and most common) case
                averaged_name =  f"{representative_pair[0]}:{resids['resna']}:{resids['resida']}-{representative_pair[1]}:{resids['resnb']}:{resids['residb']}"
            ################## Happening Inter-Subunit ######################
            else:
                # filter and catch the averaged_name if it's in to_average
                to_drop = []
       
                matched_name = None
                
                for contact_name in to_average:
                        contact_info = parse_id(contact_name)
                        # drop everything that's not from the 
                        # equivalent_interaction set

                        # The measuring requirement is here: in the current equivalent_interactions you would just have a 
                        # key pair that is not duplicated in its list value and the non sorted pairs in the list
                        # tell you implicitly which subunit a residue should come from but you can't use this information
                        # and instead have to make the distance measurements.
                        if (contact_info['chaina'],contact_info['chainb']) not \
                            in equivalent_interactions[representative_pair]:
                            to_drop.append(contact_name)
                        # if one of to_average involves both of 
                        # representative_pair's chains, then the name is 
                        # determined    
                        if (contact_info['chaina'], 
                            contact_info['chainb']) == representative_pair:
                            matched_name = contact_name

                # drop the ones that don't fit the original pair
                for contact_name in to_drop:
                        to_average.remove(contact_name)


                ########################## if inter-hetero subunit ##########    
                if get_chain_group(representative_pair[0], identical_subunits) \
                != get_chain_group(representative_pair[1], identical_subunits):
                        # and the order that the hetero subunit appears in 
                        # representative_pairs matches, you can name it without 
                        # further consideration 
                        if get_chain_group(representative_pair[0], 
                                           identical_subunits) ==  \
                                            get_chain_group(resids['chaina'], 
                                                            identical_subunits):
                            averaged_name =  f"{representative_pair[0]}:{resids['resna']}:{resids['resida']}-{representative_pair[1]}:{resids['resnb']}:{resids['residb']}"
                else: 
                        # if a contact was found earlier that matches the 
                        # representative pair, you name it now
                        if matched_name is not None:
                            averaged_name = matched_name

                        else:
                        # have to determine if the original contact for this
                        # iteration happens to have the flipped 
                        # inter-subunit naming and to_average didn't include
                        # the contact with the representative naming scheme
                        # (should not happen often)
                        # If it hasn't been figured out at this point,
                        # we measure the distance between the contacting 
                        # residues for each contact and determine the 
                        # distance that these residues should be apart
                        # and then test the representative_pair residue 
                        # distances with the default name and with the 
                        # flipped resid info

                            # measure each distance in to_average's contacts ###
                            contact_distances = []
                            for contact_name in to_average:
                                contact_info = parse_id(contact_name)
                                sel1 = f"chainID {contact_info['chaina']} and resnum {contact_info['resida']} and name CA"
                                sel2 = f"chainID {contact_info['chainb']} and resnum {contact_info['residb']} and name CA"
                                contact_distances.append(
                                    get_pair_distance(sel1, sel2, u)
                                    )

                            #### get correct contact averaged_name by checking 
                            ## two possible distances ##
                            mean_distance = np.mean(contact_distances)

                            testa1 = f"chainID {representative_pair[0]} and resnum {contact_info['resida']} and name CA"
                            testa2 = f"chainID {representative_pair[1]} and resnum {contact_info['residb']} and name CA"
                            testb1 = f"chainID {representative_pair[1]} and resnum {contact_info['resida']} and name CA"
                            testb2 = f"chainID {representative_pair[0]} and resnum {contact_info['residb']} and name CA"

                            testa = get_pair_distance(testa1, testa2, u)
                            testb = get_pair_distance(testb1, testb2, u)
                            
        # if the difference between the mean contact distance distance and the 
        # chain swapped one is greater....
                            if np.abs(testa-mean_distance) < \
                                np.abs(testb-mean_distance):
                                averaged_name = f"{representative_pair[0]}:{resids['resna']}:{resids['resida']}-{representative_pair[1]}:{resids['resnb']}:{resids['residb']}"
                            else:
                                averaged_name =  f"{representative_pair[0]}:{resids['resnb']}:{resids['residb']}-{representative_pair[1]}:{resids['resna']}:{resids['resida']}"
            ###### End flipped contact name adjustment ###########
            ################################ End inter-subunit ################
            ### At this point, contacts incorrectly captured by regex filter
            ### should be dropped, and the correct averaged name determined.

            # separate intra and inter that are occurring with the same residues
            if identical_pair:
                inter = []
                for contact_name in to_average:
                    contact_info = parse_id(contact_name)
                    if contact_info['chaina'] != contact_info['chainb']:
                        inter.append(contact_name)
                for contact in inter:
                    to_average.remove(contact)
            else:
                intra = []
                for contact_name in to_average:
                    contact_info = parse_id(contact_name)
                    if contact_info['chaina'] == contact_info['chainb']:
                        intra.append(contact_name)
                for contact in intra:
                    to_average.remove(contact)

            if len(to_average) > denominator:
                # in the case where an inter-subunit contact can happen
                # from A to B and B to A on a trimer or larger
                # you'll have more contacts than there are subunits
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
                            original_contacts:pd.DataFrame,
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
    equivalent_interactions = get_equivalent_interactions(u, 
                                                          identical_subunits,
                                                          representative_chains)


    replicated_contacts = {}
    mapped_contacts = {contact:[] for contact in averaged_contacts.columns}
    
    for contact in tqdm.tqdm(averaged_contacts.columns):
        resids = parse_id(contact)
        chaina = resids['chaina']
        chainb = resids['chainb']
        if chaina == chainb:
             for identical_subunit in identical_subunits[get_chain_group(
                                                    chaina,identical_subunits)]:
                  if identical_subunit != chaina:
                       equivalent_contact = f"{identical_subunit}:{resids['resna']}:{resids['resida']}-{identical_subunit}:{resids['resnb']}:{resids['residb']}"
                       replicated_contacts[equivalent_contact] = averaged_contacts[contact]
                       mapped_contacts[contact].append(equivalent_contact)
        else:   
            equivalent_pairs = equivalent_interactions[(chaina,chainb)]
            for equivalent_pair in equivalent_pairs:
                if equivalent_pair == (chaina, chainb):
                    replicated_contacts[contact] = averaged_contacts[contact]
                    mapped_contacts[contact].append(contact)
                    continue
                # now have to check every time to determine if the 
                # averaged_contact happens to have
                # a flipped name relative to the other chain pairs 
                # (alphabetical order swapping)
                testa = f"{equivalent_pair[0]}:{resids['resna']}:{resids['resida']}-{equivalent_pair[1]}:{resids['resnb']}:{resids['residb']}"
                testb = f"{equivalent_pair[0]}:{resids['resnb']}:{resids['residb']}-{equivalent_pair[1]}:{resids['resna']}:{resids['resida']}"
                testa_in = testa in original_contacts.columns
                testb_in = testb in original_contacts.columns
                # if both contact names are in the original data, then measure 
                # distances and compare them to the averaged contact
                # if they're hetero-subunit, can name without further consideration
                if get_chain_group(chaina, identical_subunits) != \
                                    get_chain_group(chainb, identical_subunits):
                    equivalent_contact = testa
                    replicated_contacts[equivalent_contact] = averaged_contacts[contact]
                elif testa_in and testb_in:
                    sel1 = f"chainID {chaina} and resnum {resids['resida']} and name CA"
                    sel2 = f"chainID {chainb} and resnum {resids['residb']} and name CA"
                    ref_dist = get_pair_distance(sel1, sel2, u)

                    testa1 = f"chainID {equivalent_pair[0]} and resnum {resids['resida']} and name CA"
                    testa2 = f"chainID {equivalent_pair[1]} and resnum {resids['residb']} and name CA"
                    testb1 = f"chainID {equivalent_pair[1]} and resnum {resids['resida']} and name CA"
                    testb2 = f"chainID {equivalent_pair[0]} and resnum {resids['residb']} and name CA"
                    testa_dist = get_pair_distance(testa1, testa2, u)
                    testb_dist = get_pair_distance(testb1, testb2, u)
                    # can print pairs of distances that are under some 
                    # difference threshhold
                    if np.abs(testa_dist-ref_dist) < np.abs(testb_dist-ref_dist):
                        equivalent_contact = testa
                    else:
                        equivalent_contact = testb
                    replicated_contacts[equivalent_contact] = averaged_contacts[contact]
                elif testa in original_contacts.columns:
                    equivalent_contact = testa
                    replicated_contacts[equivalent_contact] = averaged_contacts[contact]
                elif testb in original_contacts.columns:
                    equivalent_contact = testb
                    replicated_contacts[equivalent_contact] = averaged_contacts[contact]
                elif (testa not in original_contacts.columns) and (testb not in original_contacts.columns) :
                    sel1 = f"chainID {chaina} and resnum {resids['resida']} and name CA"
                    sel2 = f"chainID {chainb} and resnum {resids['residb']} and name CA"
                    ref_dist = get_pair_distance(sel1, sel2, u)

                    testa1 = f"chainID {equivalent_pair[0]} and resnum {resids['resida']} and name CA"
                    testa2 = f"chainID {equivalent_pair[1]} and resnum {resids['residb']} and name CA"
                    testb1 = f"chainID {equivalent_pair[1]} and resnum {resids['resida']} and name CA"
                    testb2 = f"chainID {equivalent_pair[0]} and resnum {resids['residb']} and name CA"
                    testa_dist = get_pair_distance(testa1, testa2, u)
                    testb_dist = get_pair_distance(testb1, testb2, u)
                    # can print pairs of distances that are under some 
                    # difference threshhold
                    if np.abs(testa_dist-ref_dist) < np.abs(testb_dist-ref_dist):
                        equivalent_contact = testa
                    else:
                        equivalent_contact = testb
                    replicated_contacts[equivalent_contact] = averaged_contacts[contact]
            
                mapped_contacts[contact].append(equivalent_contact)
 
                    
    if as_map:
         return mapped_contacts
    else:
        return pd.DataFrame.from_dict(replicated_contacts)
