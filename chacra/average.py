import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import pandas as pd
from itertools import combinations, permutations
#from .ContactFrequencies import *
from .utils import *
import tqdm
import os
from scipy.spatial.transform import Rotation as R
import re

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

def get_long_axis(subunit:mda.AtomGroup) -> np.ndarray:
    '''
    Get the vector corresponding to the long axis of the provided
    subunit atomgroup

    subunit : mda.AtomGroup
        e.g. u.segments[0].atoms
        or u.select_atoms('chainID A') etc.

    Returns
    -------
    np.ndarray
        The long axis of the atomgroup
    '''
    coords = subunit.positions - subunit.center_of_mass()
    inertia = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(inertia)  # returns sorted eigenvalues
    long_axis = eigvecs[:, -1]  # last vector = largest eigenvalue = long axis
    return long_axis

def align_local_axis_to_global_axis(local_axis:np.ndarray,
                                    global_axis:list|np.ndarray=[0,1,0]
                                    ) -> np.ndarray:
    '''
    Get the rotation matrix to align the local_axis (e.g. long axis from
    get_long_axis) to a global axis where [0,1,0] is the y-axis.

    local_axis : np.ndarray
        The vector corresponding to the long axis of an atom group
        to be aligned.

    global_axis : list | np.ndarray
        A vector with a one in the element corresponding to the axis
        that the local_axis will be aligned to.
        [1,0,0]:x, [0,1,0]:y, [0,0,1]:z

    Returns
    -------
    np.ndarray
        The rotation matrix for aligning local to global axis.
    '''
    z_axis = np.array(global_axis)
    axis = local_axis / np.linalg.norm(local_axis)
    cross = np.cross(local_axis, z_axis)
    dot = np.dot(local_axis, z_axis)
    if np.allclose(cross, 0):
        return np.eye(3) if dot > 0 else -np.eye(3)
    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    R = np.eye(3) + skew + (skew @ skew) * ((1 - dot) / (np.linalg.norm(cross) ** 2))
    return R

def align_universe_by_subunit(u:mda.Universe, 
                              subunit:mda.AtomGroup, 
                              axis:list|np.ndarray=[0,1,0]
                              ):
    '''
    In place alignment of a universe based on the alignment of a subunit
    to a specified axis.

    subunit : mda.AtomGroup
        e.g. u.segments[0].atoms
        or u.select_atoms('chainID A') etc.

    global_axis : list | np.ndarray
        A vector with a one in the element corresponding to the axis
        that the local_axis will be aligned to.
        [1,0,0]:x, [0,1,0]:y, [0,0,1]:z

    Returns
    -------
    Universe coordinates are updated in place.
    '''
    
    local_axis = get_long_axis(subunit)
    R = align_local_axis_to_global_axis(local_axis, axis)
    
    universe_com = u.atoms.center_of_mass()
    shifted = u.atoms.positions - universe_com

    rotated = shifted @ R.T

    u.atoms.positions = rotated + universe_com

def get_rotation_matrix(u:mda.Universe,
                        chaina:mda.AtomGroup, 
                        chainb:mda.AtomGroup) -> np.ndarray:
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

    align_universe_by_subunit(u, chaina)
    matrix, rmsd = align.rotation_matrix(chaina.positions, 
                                         chainb.positions)

    return matrix

def get_all_rotations(u:mda.Universe
                      ) -> dict[int, dict[int, np.ndarray]]:
    
    '''
    Get all of the rotations for subunit i to subunits j to N for all subunits.

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

    u = u.copy()
    segids = [i for i in range(len(u.segments))]
    seg_combos = [combo for combo in permutations(segids,2)]
    rotations = {segid: {segid:0 for segid in segids}
                  for segid in segids}
    for combo in seg_combos:
        sega, segb = combo
        rotations[sega][segb] = get_rotation_matrix(u,
                                                    u.segments[sega].atoms,
                                                      u.segments[segb].atoms)
    return rotations

def get_equivalent_pairs(rotations:dict)->set[tuple]:
    '''
    Returns set of tuples with pairs of tuples that represent equivalent
    rotations. {
                    ((0,1),(1,2)),
                    ((1,2),(2,0))
                }
    This shows that the rotation matrices of segment 0 to 1 and 1 to 2 are the 
    same, as well as the rotations of 1 to 2 and 2 to 0.

    rotations : The dictionary of subunit rotations from get_all_rotations().

    Returns
    -------
    set
        Pairs of segment id pairs that are equivalent in rotation
    '''
    pairs= set()
    # key will be segid
    for seg1, matrix_dict1 in rotations.items():
        # then go through all other segids and compare 
        # their rotations 
        for sega, matrixa in matrix_dict1.items():
            if sega==seg1:
                continue
            
            for seg2, matrix_dict2 in rotations.items():
                # need to handle errors here
                # segb = np.where([np.allclose(matrixa,matrixb, atol=1e-02) 
                #                  if not isinstance(matrixb, int) 
                #                  else False for 
                #                  matrixb in matrix_dict2.values()]
                #                  )[0][0]
                
                segb = np.argmin([
                (np.linalg.norm(matrixa - matrixb, ord='fro')) 
                if not isinstance(matrixb, int) # self rotations int(0)
                else np.inf for matrixb in matrix_dict2.values()])

                # can also try to compare the euler angles on the appropriate 
                # axis since this is based on align_universe_by_subunit()
                if (seg1,sega) == (seg2,segb):
                    continue
                else:
                    pairs.add(tuple(sorted(((seg1,sega),(seg2,segb)))))
    return pairs

def get_all_pair_mappings(equivalent_pairs:set
                          ) -> dict[tuple[set[tuple]]]:
    '''
    Returns the dictionary of segment id tuples and lists of segment id tuples
    with equivalent rotations.

    equivalent_pairs : set
        Output of get_equivalent_pairs

    Returns
    -------
    dict
    '''
    mapping = {}
    for pair in equivalent_pairs:
        mapping[pair[0]] = set()
        mapping[pair[1]] = set()
    for pair in equivalent_pairs:
        mapping[pair[0]].add(pair[1])
        mapping[pair[1]].add(pair[0])
    return {key:val for key,val in 
            # the old get_equivalent_interactions() also sorted the order
            # of the tuple contents which does not preserve the relationship
            # e.g. {('A','B'): [('B','C'),...('A','D')]} when that last
            # element is more accurately ('D','A') - have to handle
            # in the averaging function
            sorted(mapping.items(),key=lambda item: item[0])}

def make_pair_mapping_with_chainids(pair_mapping: dict,
                                    u: mda.Universe
                                    ) -> dict:
    '''
    Returns the same subunit pair rotation mapping but using chainID strings
    
    '''
    if not [set(seg.atoms.chainIDs).pop() for seg in u.segments] == \
        list(u.segments.segids):
        print("Segment and chainIDs don't match. "\
              "Set the chainIDs and segment ids to the same value as the "\
                "chainIDs in the Universe before continuing.")
        return 
    else:
        # seg index to chain/seg ID
        sm = {i:segid for i,segid in enumerate(u.segments.segids)}
        chain_mapping = {}
        for pair, tup_list in pair_mapping.items():
            chain_mapping[(sm[pair[0]], sm[pair[1]])] = \
            list([(sm[tup[0]], sm[tup[1]]) for tup in tup_list])
        
    return chain_mapping

def get_pair_mappings(u:mda.Universe) -> dict[tuple[set[tuple]]]:
    '''
    Takes a symmetric homomultimer structure and returns a dictionary
    of each subunit pair and a set of all the equivalent subunit pairings.
    e.g. if B is to the right of A and C is to the right of B and so on, the 
    mapping is ('A','B'):{('B','C'),...}
    
    Parameters
    ----------
    u: mda.Universe
        A universe containing a symmetric homomultimer.

    Returns
    -------
    dict
        The mappping of subunit pairs to the equivalent subunit pairings.
    '''
    rotations = get_all_rotations(u)
    equivalent_pairs = get_equivalent_pairs(rotations)
    pair_mappings = get_all_pair_mappings(equivalent_pairs)
    return make_pair_mapping_with_chainids(pair_mappings, u)


def find_identical_subunits(universe):
    '''
    
    determines which subunits are identical

    Parameters
    ----------
    universe: 
        mda.Universe from the structure that the contact
        data was calculated on.
    
    Returns
    -------
    Dictionary containing integer id keys and lists of identical subunits ids.
    '''
    # This is overly complex

    # dictionary of segids: list/sequence of residues
    residues = {seg.segid: seg.residues.resnames for seg in universe.segments}
    segids = list(residues.keys())
    # make a square matrix that will be filled with True values 
    # for identical subunits
    array = np.zeros((len(segids),len(segids)),dtype=np.bool_)
    # every subunit is identical with itself
    np.fill_diagonal(array,True)
    # work with it as a df
    identical_table = pd.DataFrame(array, columns=segids, index=segids)

    # Go through all pairwise combinations of subunits
    for combo in combinations(segids,2):
        # not identical if lengths are different
        if len(residues[combo[0]]) != len(residues[combo[1]]):
            identical_table[combo[1]][combo[0]], identical_table[combo[0]][combo[1]] = False, False
        else:
            # catch anything that might have same number of residues but 
            # different sequences
            bool = np.all(np.equal(residues[combo[0]],residues[combo[1]]))
            # Enter True or False in both the upper tri and lower tri  
            identical_table[combo[1]][combo[0]], identical_table[combo[0]][combo[1]] =  bool, bool

    # Only keep one representative row for each unique chain that gives the 
    # identical sets of subunits
    identical_table.drop_duplicates(inplace=True)

    identical_subunits = {}
    # Add the lists of identical subunits to the dictionary
    # You can deal with different sets of identical subunits in complex situations\
    for i, segid in enumerate(identical_table.index):
        subunits = identical_table.T[identical_table.loc[segid]==True].index
        # if there is only one, it's just identical with itself
        if len(subunits) >= 2:
            identical_subunits[i] = sorted(list(subunits))

    return identical_subunits

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

def get_equivalent_interactions(u, 
                                identical_subunits,
                                representative_chains):
    
    # I think it's better to maintain the unsorted tuple contents since it's
    # a more accurate description of the subunit relationships and could
    # be useful later
    # THUS
    # run get_pair_mappings on universes containing only
    # a single type of subunit and then combine
    equivalent_interactions = {}
    seperated_subunits = {}
    # taking each list of identical subunits
    for subunits in identical_subunits.values():
        # make an MDAnalysis selection of the identical subunits
        sel_str = "chainID "
        for subunit in subunits[:-1]:
            sel_str+=f"{subunit} or chainID "
        sel_str += subunits[-1]
        # Then make a universe with and put it in a dictionary
        # and associate it with the representative chain.
        for rch in representative_chains:
            if rch in subunits:
                seperated_subunits[rch] = mda.Merge(
                            u.select_atoms(sel_str))
    # then get the equivalent interaction dictionary
    # and merge with the equivalent interactions of the other
    # ones built on identical subunits
    for rch, su in seperated_subunits.items():           
        pair_mappings = get_pair_mappings(su)
        # to be compatible with with the old version, need to only have the 
        # tuples with the representative chain and also add that tuple to its own list
        # and then when accessing them, they have to be sorted() 
        equivalent_interactions |= {tuple(sorted(key)):
                    [tuple(sorted(tup)) for tup in val] + [tuple(sorted(key))]
                                    for key, val in pair_mappings.items()
                                    if key[0] == rch}
    return equivalent_interactions

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

def get_rotation_matrix_euler_angles(matrix):
    '''
    Not used
    '''
    r = R.from_matrix(matrix)

    # Get Euler angles in degrees or radians
    euler_angles_rad = r.as_euler('xyz', degrees=False)
    euler_angles_deg = r.as_euler('xyz', degrees=True)
    return euler_angles_deg

def find_identical_subunits(universe):
    '''
    
    determines which subunits are identical

    Parameters
    ----------
    universe: 
        mda.Universe from the structure that the contact
        data was calculated on.
    
    Returns
    -------
    Dictionary containing integer id keys and lists of identical subunits ids.
    '''
    # This is overly complex

    # dictionary of segids: list/sequence of residues
    residues = {seg.segid: seg.residues.resnames for seg in universe.segments}
    segids = list(residues.keys())
    # make a square matrix that will be filled with True values 
    # for identical subunits
    array = np.zeros((len(segids),len(segids)),dtype=np.bool_)
    # every subunit is identical with itself
    np.fill_diagonal(array,True)
    # work with it as a df
    identical_table = pd.DataFrame(array, columns=segids, index=segids)

    # Go through all pairwise combinations of subunits
    for combo in combinations(segids,2):
        # not identical if lengths are different
        if len(residues[combo[0]]) != len(residues[combo[1]]):
            identical_table[combo[1]][combo[0]], identical_table[combo[0]][combo[1]] = False, False
        else:
            # catch anything that might have same number of residues but 
            # different sequences
            bool = np.all(np.equal(residues[combo[0]],residues[combo[1]]))
            # Enter True or False in both the upper tri and lower tri  
            identical_table[combo[1]][combo[0]], identical_table[combo[0]][combo[1]] =  bool, bool

    # Only keep one representative row for each unique chain that gives the 
    # identical sets of subunits
    identical_table.drop_duplicates(inplace=True)

    identical_subunits = {}
    # Add the lists of identical subunits to the dictionary
    # You can deal with different sets of identical subunits in complex situations\
    for i, segid in enumerate(identical_table.index):
        subunits = identical_table.T[identical_table.loc[segid]==True].index
        # if there is only one, it's just identical with itself
        if len(subunits) >= 2:
            identical_subunits[i] = sorted(list(subunits))

    return identical_subunits

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

#################### The Main Function ##########################
#################### It's a Beast ##########################


def average_multimer(structure: str|os.PathLike, 
                     df: pd.DataFrame, 
                     representative_chains: list=None,
                     return_stdev:bool=False, 
                     check_folder='equivalent_interactions'
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
            else:
                validate_group_memberships(representative_chains, 
                                           identical_subunits)
    equivalent_interactions = get_equivalent_interactions(u,
                                                        identical_subunits,
                                                        representative_chains)
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


def everything_from_averaged(averaged_contacts, original_contacts, u, 
                             representative_chains,
                             as_map=False):
    '''
    Take the averaged contacts and regenerate the entire protein's contacts 
    using this data. Useful for visualizing the flow of chacras across the whole 
    protein and doing network analysis on more statistically robust data.

    averaged_contacts : pd.DataFrame
        The averaged contact data

    u : MDA.Universe 
        The same universe used for averaging contacts

    representative_chains : list of strings
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
        return replicated_contacts

#### For depicting the high loading score contacts on all subunits,
## return as_map, and provide the original averaged high loading score contacts
# and the map to pymol_averaged_chacras_to_all_subunits
#


############## Deprecated #################################




# def find_best_asymmetric_point(u, chain, all_chain_dists):
#     '''
#     Find the residue that creates the greates difference in distances between 
#     other chains by finding a point in the chain that's near some other 
#     neighboring chain.

#     #TODO can also sample random points on the chain and take the one
#      with the best differences
#     '''
#     A = np.where(u.atoms.segids == f'{chain}')[0]
#     # asymmetric center of mass that is some point on the periphery of seg_combo[0] near seg_combo[1]
#     # if they're not within 5 angstroms, this won't work - need to just identify closest chain, pick one at random
#     # and then establish the as_com with that
#     minimum_difference = {}
#     for neighbor, distance in all_chain_dists[chain].items():
#         if distance < 5:
#             distances_to_chains = []
#             as_com = u.select_atoms(f'(around 5 chainID {neighbor}) and chainID {chain}').center_of_mass()
#             # the distances between all the atoms in seg_combo[0] and the asymmetric center of mass
#             dists = np.apply_along_axis(np.linalg.norm, 1,u.atoms[A].positions - as_com)
#             #find the closest res to the as_com so that this residue can be used for the reference point for other identical chains
#             resid = u.atoms[A][np.where(dists==dists.min())[0][0]].resid
#             A_pos = u.select_atoms(f'chainID {chain} and resnum {resid} and name CA').positions[0]
#             for seg in u.segments.segids:
#                 if seg != chain:
#                     B_com = u.select_atoms(f'chainID {seg}').center_of_mass()
#                     distances_to_chains.append(np.linalg.norm(A_pos-B_com))
#             minimum_difference[neighbor] = min([np.abs(i-j) for i, k in enumerate(distances_to_chains)
#                                                 for j, l in enumerate(distances_to_chains) if k !=l])
    
#     return max(minimum_difference, key=minimum_difference.get)

# def asymmetric_measurements(seg_combo:tuple[str, str], 
#                             identical_subunits:dict, 
#                             u:mda.Universe, 
#                             all_chain_dists:dict) -> list[tuple[str,str]]:
#     '''

#     This will identify all of the interaction pairs equivalent to seg_combo
#     all_chain_dists is included so the best residue is used to create the best
#       variance in the measured distances 
    
#     seg_combo : tuple(string, string)
#         The pair of chains that you want to identify all the equivalent 
#         interacting pairs for. ('A','B') will find other identical chains pairs 
#         that interact with the same geometry as A and B
    
#     identical_subunits : dictionary
#         The dictionary with integer values and lists of identical subunits.

#     all_chain_dists : dictionary
#         The dictionary with chain_id values and nest dictionaries giving the 
#         chain keys and the minimum distance to them as values
    
#     Returns
#     -------
#     list of tuples
#     each tuple will be a pair of chains with equivalent interaction geometry
    
#     '''

#     # whole protein center of mass
#     com = u.select_atoms('protein').center_of_mass()
#     # indices of atoms corresponding to chain seg_combo[0]
#     A = np.where(u.atoms.segids == f'{seg_combo[0]}')[0]
#     # asymmetric center of mass that is some point on the periphery of 
#     # seg_combo[0] near seg_combo[1]
#     # if they're not within 5 angstroms, this won't work - need to just identify
#     # closest chain, pick one at random and then establish the as_com with that
#     neighbor = find_best_asymmetric_point(u, seg_combo[0], all_chain_dists)
#     as_com = u.select_atoms(f'(around 5 chainID {neighbor}) and chainID ' \
#                             f'{seg_combo[0]}').center_of_mass()
#     # the distances between all the atoms in seg_combo[0] and the asymmetric 
#     # center of mass
#     dists = np.apply_along_axis(np.linalg.norm, 1,u.atoms[A].positions - as_com)
#     #find the closest res to the as_com so that this residue can be used for the
#     # reference point for other identical chains
#     resid = u.atoms[A][np.where(dists==dists.min())[0][0]].resid
#     # better to just use CA instead of this atom for distance measurements since 
#     #there's more variability in sidechain positions
#     atom = u.atoms[A][np.where(dists==dists.min())[0][0]].name
#     A_pos = u.select_atoms(f'chainID {seg_combo[0]} and resnum {resid} and name CA'
#                            ).positions[0]
#     # center of mass of seg_combo[1]
#     B_com = u.select_atoms(f'chainID {seg_combo[1]}').center_of_mass()
#     # this identifies A's relationship to B. 
#     comparison_dist = np.linalg.norm(A_pos-B_com)
#     comparison_angle = np.rad2deg(get_angle(A_pos,com,B_com))
     
#     for key, seg_list in identical_subunits.items():
#         if seg_combo[0] in seg_list and seg_combo[1] in seg_list:
#             A_group, B_group = key, key
#         elif seg_combo[0] in seg_list and seg_combo[1] not in seg_list:
#             A_group = key
#         elif seg_combo[1] in seg_list and seg_combo[0] not in seg_list:
#             B_group = key

#     relationships = []
#     for seg1 in identical_subunits[A_group]:
#         distances = {}
#         distance_difs = {}
#         angle_difs = {}
#         for seg2 in identical_subunits[B_group]:
#             if seg1 != seg2:
            
#                 pos1 = u.select_atoms(f'chainID {seg1} and resnum {resid} and name CA'
#                                       ).positions[0]
#                 pos2 = u.select_atoms(f'chainID {seg2}').center_of_mass()
#                 distance = np.linalg.norm(pos1-pos2)
#                 distances[(seg1,seg2)] = distance
#                 angle_difs[(seg1,seg2)] = np.abs(np.rad2deg(
#                     get_angle(pos1,com,pos2)) - comparison_angle)
#                 distance_difs[(seg1,seg2)] = np.abs(distance-comparison_dist)

#         min_dist, min_angle = min(distance_difs, key=distance_difs.get), min(
#                                                 angle_difs, key=angle_difs.get)  
#         if min_dist != min_angle:
#             # sorting everything to ensure alphabetical order will align with 
#             #contact naming scheme
#             if ((distance_difs[min_dist]) + (angle_difs[min_dist])
#                 ) < (distance_difs[min_angle]) + (angle_difs[min_angle]):
#                 relationships.append(tuple(sorted(min_dist)))
#             else:
#                 relationships.append(tuple(sorted(min_angle)))
#         else:       
#             relationships.append(tuple(sorted(min_dist)))
#     return relationships

# def get_equivalent_interactions(representative_chains:list[str], 
#                                 u:mda.Universe) -> dict:
#     Deprecated
#     '''
#     For each chain in representative_chains, find all other identical chains' 
#     interaction partners that are equivalent to the representative_chains 
#     interactions with all the other chains in the protein. For instance, chain 
#     A's interaction with a subunit D on the other side of the protein might be
#     equivalent to chain B's interaction with subunit E for a symmetric multimer.

#     This is useful when averaging the contact frequencies of a multimer and 
#     determining the correct naming for the averaged contact record and to 
#     ensure it's depicted correctly when visualized.

#     Parameters
#     ----------
#     representative_chains : list of strings
#         A list with the names of the chain ids that you want to establish 
#         equivalent interactions for

#     '''

#     segids = u.segments.segids
#     all_chain_dists = get_all_chain_dists(u)
#     identical_subunits = find_identical_subunits(u)
#     # deals with homodimer
#     if len(identical_subunits) == 1 and len(segids) == 2:
#         equivalent_interactions = {tuple(sorted((segids[0],segids[1])))
#                                  : [tuple(sorted((segids[0],segids[1])))]
#                                     }
#     else:
#         equivalent_interactions = {}
#         for chain in [chain for chain in representative_chains]:
#             for segid in segids:
#                 if segid != chain:
#                     equivalent_interactions[tuple(sorted((chain,segid)))
#                                             ] = asymmetric_measurements(
#                                             (chain,segid),identical_subunits,
#                                             u, all_chain_dists)

#     return equivalent_interactions

# def equivalent_interactions_check(equivalent_interactions, 
#                                   output_folder='equivalent_interactions'):
#     '''
#     Confirm that the averaging is taking place between the appropriate subunits
#     with a series of pymol selections.

#     The pair names are in alphabetical order so the ordering of the letter
#     does not necessarily correspond to the interaction order of the original 
#     pair that each file is named after.
#     '''
#     if os.path.exists(output_folder):
#         pass
#     else:
#         os.makedirs(output_folder)

#     for pair, equivalent_pair_list in equivalent_interactions.items():
#         with open(f'{output_folder}/chains_{pair[0]}-{pair[1]}_check.pml','w') as f:
#             for equiv_pair in equivalent_pair_list:
#                 selection_string = f"select {equiv_pair[0]}-{equiv_pair[1]}, "\
#                         f"chain {equiv_pair[0]} or chain {equiv_pair[1]}"
#                 f.write(f'{selection_string} \n')

# def get_all_chain_dists(u):
#     '''
#     Minimum distances between each chain and all others
#     This takes several seconds so run it before passing output to other functions
    
#     '''

#     segids = {segid for segid in u.segments.segids}
#     sorted_all_chain_dists = {}
#     all_chain_dists = {chain:{} for chain in segids}
#     for chain in segids:
#         sel1 = u.select_atoms(f'chainID {chain}')
#         for chain2 in segids:
#             if chain2 != chain:
#                 sel2 = u.select_atoms(f'chainID {chain2}')
#                 min_dist = distance_array(sel1.atoms, sel2.atoms,).min()
#                 all_chain_dists[chain][chain2] = min_dist
#         sorted_all_chain_dists[chain]={k:v for k, v in sorted(all_chain_dists[chain].items(), key=lambda x:x[1])}

#     return sorted_all_chain_dists
