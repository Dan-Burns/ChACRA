

# break up average contacts into manageable functions
from itertools import combinations
import pandas as pd
import numpy as np
import re
from ChACRA.ContactAnalysis.contact_functions import _parse_id
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_dihedrals


 
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
    # dictionary of segids: list/sequence of residues
    residues = {seg.segid: seg.residues.resnames for seg in universe.segments}
    segids = list(residues.keys())
    # make a square matrix that will be filled with True values for identical subunits
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
            # catch anything that might have same number of residues but different sequences
            bool = np.all(np.equal(residues[combo[0]],residues[combo[1]]))
            # Enter True or False in both the upper tri and lower tri  
            identical_table[combo[1]][combo[0]], identical_table[combo[0]][combo[1]] =  bool, bool

    # Only keep one representative row for each unique chain that gives the identical sets of subunits
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

def get_all_subunit_contacts(identical_subunits, df):
    '''
    using in 1st average_contacts function

    return a list of contacts that involve at least one subunit in subunits
    
    Parameters
    ----------
    subunits: list

    '''

    subunits_string = '|'.join(identical_subunits)
    # TODO only need one residue's subunit to match
    # need to do 2 regexs to get anything where chain A matches then any leftovers where chain B
    # need to account for multi letter chains in the variable 
    regex1 = f"(?:{subunits_string}):[A-Z]{{3}}:\d+-[A-Z1-9]+:[A-Z]{{3}}:\d+" 
    regex2 = f"[A-Z1-9]+:[A-Z]{{3}}:\d+-(?:{subunits_string}):[A-Z]{{3}}:\d+"
    columns1 = list(df.filter(regex=regex1, axis=1).columns)
    columns2 = list(df.filter(regex=regex2, axis=1).columns)
    columns = columns1 + columns2
    
    return list(set(columns))


def filter_by_chain(contact_list, same_chain=False):
    '''
    Using in 1st average_contacts function

    Take a list of contacts and return contacts that:
    have the same chain id for each residue in the contact: same_chain = True
    or have different chain ids for each residue in the contact: same_chain = False (default)

    Parameters
    ----------
    contact_list: list of contacts 

    same_chain: bool 
        return a list of contacts that have different or same
          chain ids between the contact residue pairs. Default is False and will return
          a list of contacts that have different chain ids.

    Returns:
    --------
    list of contacts with same or different chain ids (determined by same_chain)
    '''
    # named to_remove because this is used in the average_contacts function to 
    # filter out the contacts that won't be included in an iteration average
    to_remove = []
    for pair in contact_list:
        check = _parse_id(pair)
        if same_chain == False:
            if check['chaina'] != check['chainb']:
                to_remove.append(pair)
        else:
            if check['chaina'] == check['chainb']:
                to_remove.append(pair)
    
    return to_remove

def get_standard_average(df, to_average, identical_subunits, check=True):
    '''
    
    deal with lists of contacts that may not equal the length of the identical
    subunit list.  If a contact is only picked up in 2 of 3 total subunits for instance,
    this will divide the sum of the two contact frequencies by 3

    Parameters
    ----------
    df: pd.DataFrame
        contact dataframe.
    
    to_average: list 
        list of contacts to include in the averaging.

    identical_subunits: list
        list of subunits whose contacts are to be included in the averaging

    check: bool
        If True (default), return the average value even if there are more contacts in to_average
        than there are subunits in identical_subunits
        If False, return nothing and print a message that the contacts were not averaged
        

    Returns
    -------
    float
    Average value of the identical residue pairs' contact frequency.


    averaged_data[contact] = get_standard_average(df, to_average, identical_subunits)
    '''

    if len(to_average) == len(identical_subunits):
        return df[to_average].mean(axis=1)
    elif len(to_average) < len(identical_subunits):
        return df[to_average].sum(axis=1)/len(identical_subunits)
    elif len(to_average) > len(identical_subunits):
        if check:
            print(f'Stopping after getting more contacts than identical subunits for contact matching {to_average[0]}: {to_average}.\
                  This can happen if a residue makes contact with multiple subunits.  Set check=False if this make sense for your system.')
            return 
        else:
            print(f'Got more contacts than identical subunits for contact matching {to_average[0]}. Maybe this is in a channel pore?')
            return df[to_average].mean(axis=1)

def get_angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle
        
def get_opposing_subunits(subunits, u):
    '''
    
    
    '''
    
    segids = set(u.segments.segids)
    seg_sels = {seg: u.select_atoms(f'segid {seg}') for seg in segids if seg in subunits} 
    # center of mass for each protomer
    coms = {segid: seg_sels[segid].center_of_mass() for segid in segids}
    all_seg_sel_string = ''.join([f'segid {seg} or ' for seg in list(segids)[:-1]])
    all_seg_sel_string += f'segid {list(segids)[-1]}'
    # center of mass for combined identical protomers
    all_com = u.select_atoms(all_seg_sel_string).center_of_mass()
    opposing_subunits = []
    check = []
    for seg in segids:
        for seg2 in [segid for segid in segids if segid != seg]:
            # segids will be in check if they have already been paired up with their opposing subunit
            if seg2 not in check:
                # measuring angle between two subunits with the combined (not just the two) com as the vertex
                # if the angle is ~180 degrees then they're opposing
                print(seg,seg2)
                print(np.rad2deg(get_angle(coms[seg],all_com,coms[seg2])))
                if np.abs(180-np.rad2deg(get_angle(coms[seg],all_com,coms[seg2]))) <= 2:
                    ids = [seg,seg2]
                    ids.sort()
                    opposing_subunits.append((tuple(ids)))
                    check.extend(tuple(ids))
    return opposing_subunits


def get_opposing_subunit_contacts(to_average, opposing_subunits):

    '''
    Using A/C B/D default for test
    
    '''
    opposing_contacts = []
    # if the name matches an opposing pair
    for pair in to_average:
        for subunits in opposing_subunits:
            if (subunits[0] in re.split(':|-', pair)) and (subunits[1] in re.split(':|-', pair)):

                # record it 
                opposing_contacts.append(pair)
    return opposing_contacts


def get_non_identical_subunit_contacts(df, identical_subunits):
    '''
    Return a filtered set of column names involving non-identical subunit contacts
    '''

    non_identical_subunit_sets = {}
    combos = combinations(identical_subunits.keys(),2)
    for combo in combos:
        regex1 = f"[{'|'.join(identical_subunits[combo[0]])}]:[A-Z]+:[1-9]+-[{'|'.join(identical_subunits[combo[1]])}]:[A-Z]+:[1-9]+"
        regex2 = f"[{'|'.join(identical_subunits[combo[1]])}]:[A-Z]+:[1-9]+-[{'|'.join(identical_subunits[combo[0]])}]:[A-Z]+:[1-9]+"
        regex = f"{regex1}|{regex2}"
    #this will have the identical_subunit key pairs and corresponding column names
    # you can create the regex using these keys and identical_subunits dictionary
    non_identical_subunit_sets[combo] = list(df.filter(regex=regex, axis=1).columns)
    return non_identical_subunit_sets



#####################################
'''
Need to pick a priority chain to base all the naming off
If no priority chain provided, go alphabetic priority
So specify the priority chain and then caclulate the distance to the other chains
then sort then by distance (for non identical sets too)

(Distance Matrix to centroid of chains or closest residue pair - go alphabetic if 2 are same distance)

then as a new contact is encountered, you'll have a list of all the identical contacts after filtering
will have to measure the distances to determine if this follows a closest neighbor pattern or another pattern
and use the naming scheme for the priority chain and the adjacent chain that has a similar distance as the
example contact.

'''
def get_chain_distances(identical_subunits,u):
    '''
    Determine which subunits are closest together in order to set up the averaged contact naming scheme
    '''

    # can make a table of the equivalent contacts...
    sorted_distances = {}

    #for identical_subunits_key in identical_subunits
    # find distances between identical and non identical and for all non identical, choose the closest one to priority as the rep for those
    for key in identical_subunits:
        chain_distances = {}
        priority = identical_subunits[key][0]
        sel1 = u.select_atoms(f'chainID {priority}')
        for chain in identical_subunits[key][1:]:
            sel2 = u.select_atoms(f'chainID {chain}')

            min_dist = distance_array(sel1.atoms, sel2.atoms,).min()
            chain_distances[(priority, chain)] = min_dist
        sorted_distances[key]={k:v for k, v in sorted(chain_distances.items(), key=lambda x:x[1])}

    # then get mixed subunit distances
    # this should possibly go after determining the priority chain
    # but before determing the other identical subunit priority naming since it will find which of the 
    # non identical subunits is closest....
    for combo in combinations(identical_subunits.keys(),2):
        chain_distances = {}
        priority = identical_subunits[combo[0]][0]
        sel1 = u.select_atoms(f'chainID {priority}')
        for chain in identical_subunits[combo[1]]:
            sel2 = u.select_atoms(f'chainID {chain}')

            min_dist = distance_array(sel1.atoms, sel2.atoms,).min()
            chain_distances[(priority, chain)] = min_dist
        sorted_distances[combo]={k:v for k, v in sorted(chain_distances.items(), key=lambda x:x[1])}

    # After determining the priority nameing scheme, contact distances can be checked to find the best partner subunit with allclose(2 decimals)
    # in this case - intra is going to be A-A
    # adjacent will be A-C, if it also occurs AB use all close to determine if it's equivalent 
    return sorted_distances

def get_contacting_chains(df):
    '''
    Use the get contacts data to identify which chains actually make contacts.

    '''
    partner_chains = {}

    for contact in df.columns:
        data = _parse_id(contact)
        if data['chaina'] not in partner_chains:
            partner_chains[data['chaina']] = set(data['chainb'])
        else:
            partner_chains[data['chaina']].add(data['chainb'])
        if data['chainb'] not in partner_chains:
            partner_chains[data['chainb']] = set(data['chaina'])
        else:
            partner_chains[data['chainb']].add(data['chaina'])
    
    return  partner_chains

def get_chain_group(chain, identical_subunits):
    '''
    Return the identical subunit group that the subunit is in
    '''
    for group in identical_subunits:
        if chain in identical_subunits[group]:
            return group
        
def get_chain_tuples(group1,group2):
    '''
    Gets the key id to use with the sorted_distances to use in searching
    for closest pair involving priority names
    '''
    if group1 != group2:
        return tuple(sorted((group1,group2)))
    else:
        return group1

def check_distances(group, contact, u, sorted_distances):

    dists = {}
    chain_pairs = list(sorted_distances[group].keys())
    resids = _parse_id(contact)
    for pair in chain_pairs:
        # for non identical subunits, got to make sure the right chain goes with the
        # right residue number
        # should also construct the reverse contact naming for identical subunits
        # and settle on a way of preferentially choosing an A-B contact over C-A
        # 
        atoma = u.select_atoms(f'chainid {pair[0]} and resnum \
                            {resids["resida"]} and name CA').positions
        atomb = u.select_atoms(f'chainid {pair[1]} and resnum \
                            {resids["residb"]} and name CA').positions
        
        ##### Test ##############
        atomc = u.select_atoms(f'chainid {pair[1]} and resnum \
                            {resids["resida"]} and name CA').positions
        atomd = u.select_atoms(f'chainid {pair[0]} and resnum \
                            {resids["residb"]} and name CA').positions


        ###End Test ################
        dists[f'{pair[0]}:{resids["resna"]}:{resids["resida"]}-{pair[1]}:{resids["resnb"]}:{resids["residb"]}'] = np.linalg.norm((atoma-atomb)) 
        ### 
        dists[f'{pair[1]}:{resids["resna"]}:{resids["resida"]}-{pair[0]}:{resids["resnb"]}:{resids["residb"]}'] = np.linalg.norm((atomc-atomd))
    distances = {k:v for k, v in sorted(dists.items(), key=lambda x:x[1])}
    return distances
    # returns the priority subunit names to use
    #return list(distances.keys())[0]

# def find_non_matching_angles(reference_contact, angles, cutoff=1):
#     '''
#     Return a list of contacts whose angles don't match the reference contact

#     reference_contact : string
#         The contact name whose angle will be used for comparison with the other angles.

#     angles : dict
#         The dictionary of contacts and corresponding angles 

#     cutoff : float or int
#         The maximum difference between reference_contact's angle and the comparison angle to be considered the same.
#         Contacts that differ in angle above this value will be returned in the list.

#     '''
#     non_matching_contacts = []

#     for contact in angles:
#         if np.abs(angles[reference_contact]-angles[contact]) > cutoff:
#             non_matching_contacts.append(contact)

#     return non_matching_contacts

def get_all_chain_dists(u):
    '''
    Minimum distances between each chain and all others
    This takes several seconds so run it before passing output to other functions
    
    '''

    segids = {segid for segid in u.segments.segids}
    sorted_all_chain_dists = {}
    all_chain_dists = {chain:{} for chain in segids}
    for chain in segids:
        sel1 = u.select_atoms(f'chainID {chain}')
        for chain2 in segids:
            if chain2 != chain:
                sel2 = u.select_atoms(f'chainID {chain2}')
                min_dist = distance_array(sel1.atoms, sel2.atoms,).min()
                all_chain_dists[chain][chain2] = min_dist
        sorted_all_chain_dists[chain]={k:v for k, v in sorted(all_chain_dists[chain].items(), key=lambda x:x[1])}

    return sorted_all_chain_dists


def get_farthest_point(seg,u):
    '''
    get farthest point of seg from center of mass of entire protein complex
    '''
    com = u.select_atoms('protein').center_of_mass()
    # get the indices of the atoms from each chain
    A = np.where(u.atoms.segids == f'{seg}')[0]
    
    # calculate all of the distances beteen the atoms from each chain
    daA = distance_array(u.atoms[A], com)

    farthest_distance_A = daA.max()
    
    
    # find the index of the minimum distance value
    Amax = np.where(daA == farthest_distance_A)[0]
    # get the residue id and atom name for the closest distance atoms
    resa, atoma = u.atoms[A][Amax[0]].resid, u.atoms[A][Amax[0]].name
    return resa, atoma
    
def get_dihedral_between_chains(sel1, sel2, u):
    #pos1 = u.select_atoms(f'chainID {sel1[0]} and resnum {sel1[1]} and name {sel1[2]}').positions[0]
    pos1 = u.select_atoms(f'chainID {sel1[0]} and resnum {sel1[1]} and name CA').positions[0]
    com1 = u.select_atoms(f'chainID {sel1[0]}').center_of_mass()
    #pos2 = u.select_atoms(f'chainID {sel2[0]} and resnum {sel2[1]} and name {sel2[2]}').positions[0]
    pos2 = u.select_atoms(f'chainID {sel2[0]} and resnum {sel2[1]} and name CA').positions[0]
    com2 = u.select_atoms(f'chainID {sel2[0]}').center_of_mass()
    result = np.rad2deg(calc_dihedrals(pos1,com1,com2,pos2))
    return result
####################### Below is working #################################

def make_equivalent_contact_regex(resids):
     '''
     resids : the _parse_id dictionary containing the contact data
     '''
     regex1 = f"[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)-[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)"
     regex2 = f"[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)-[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)"
     return f"{regex1}|{regex2}"

def get_representative_pair_name(chaina, chainb, identical_subunits, representative_chains, equivalent_interactions):
     '''
     provide the chains from the contact and get the chain names to use for making a generalized/averaged contact name
     '''
     if chaina == chainb:
          for key, identical_subunit_list in identical_subunits.items():
               if chaina in identical_subunit_list:
                    for representative_chain in representative_chains:
                         if representative_chain in identical_subunit_list:
                              representative_pair = (representative_chain, representative_chain)
                                   
     # determine which equivalent_interaction set it came from 
     else:
          paira = (chaina,chainb)
          
          for representative_pair_name, equivalent_interaction_list in equivalent_interactions.items():
               for pair in equivalent_interaction_list:
                    if paira == pair: #or pairb == pair:
                         representative_pair = representative_pair_name
                         
                         break
     
     return representative_pair

def get_pair_distance(sel1, sel2, u):
    '''
    Return the distance between two selections
    '''
    a = u.select_atoms(sel1).positions[0]
    b = u.select_atoms(sel2).positions[0]
    return np.linalg.norm(a-b)

def find_best_asymmetric_point(u, chain, all_chain_dists):
    '''
    Find the residue that creates the greates difference in distances between other chains
    by finding a point in the chain that's near some other neighboring chain

    #TODO can also just sample N=20? random points on the chain taken from evenly spaced points in 3D
        and take the one with the best differences
    '''
    A = np.where(u.atoms.segids == f'{chain}')[0]
    # asymmetric center of mass that is some point on the periphery of seg_combo[0] near seg_combo[1]
    # if they're not within 5 angstroms, this won't work - need to just identify closest chain, pick one at random
    # and then establish the as_com with that
    minimum_difference = {}
    for neighbor, distance in all_chain_dists[chain].items():
        if distance < 5:
            distances_to_chains = []
            as_com = u.select_atoms(f'(around 5 chainID {neighbor}) and chainID {chain}').center_of_mass()
            # the distances between all the atoms in seg_combo[0] and the asymmetric center of mass
            dists = np.apply_along_axis(np.linalg.norm, 1,u.atoms[A].positions - as_com)
            #find the closest res to the as_com so that this residue can be used for the reference point for other identical chains
            resid = u.atoms[A][np.where(dists==dists.min())[0][0]].resid
            A_pos = u.select_atoms(f'chainID {chain} and resnum {resid} and name CA').positions[0]
            for seg in u.segments.segids:
                if seg != chain:
                    B_com = u.select_atoms(f'chainID {seg}').center_of_mass()
                    distances_to_chains.append(np.linalg.norm(A_pos-B_com))
            minimum_difference[neighbor] = min([np.abs(i-j) for i, k in enumerate(distances_to_chains)
                                                for j, l in enumerate(distances_to_chains) if k !=l])
    
    return max(minimum_difference, key=minimum_difference.get)

def asymmetric_measurements(seg_combo, identical_subunits, u, all_chain_dists):
    '''
    This will identify all the equivalent interaction pairs as seg_combo
    all_chain_dists is included so the best residue is used to create the best variance in the measured distances 
    
    seg_combo : tuple(string, string)
        The pair of chains that you want to identify all the equivalent interacting pairs for.
        ('A','B') will find other identical chains pairs that interact with the same geometry as A and B
    
    identical_subunits : dictionary
        The dictionary with integer values and lists of identical subunits.

    all_chain_dists : dictionary
        The dictionary with chain_id values and nest dictionaries giving the chain keys and the minimum distance to them as values
    
    Returns
    -------
    list of tuples
    each tuple will be a pair of chains with equivalent interaction geometry
    
    '''

    # whole protein center of mass
    com = u.select_atoms('protein').center_of_mass()
    # indices of atoms corresponding to chain seg_combo[0]
    A = np.where(u.atoms.segids == f'{seg_combo[0]}')[0]
    # asymmetric center of mass that is some point on the periphery of seg_combo[0] near seg_combo[1]
    # if they're not within 5 angstroms, this won't work - need to just identify closest chain, pick one at random
    # and then establish the as_com with that
    neighbor = find_best_asymmetric_point(u, seg_combo[0], all_chain_dists)
    as_com = u.select_atoms(f'(around 5 chainID {neighbor}) and chainID {seg_combo[0]}').center_of_mass()
    # the distances between all the atoms in seg_combo[0] and the asymmetric center of mass
    dists = np.apply_along_axis(np.linalg.norm, 1,u.atoms[A].positions - as_com)
    #find the closest res to the as_com so that this residue can be used for the reference point for other identical chains
    resid = u.atoms[A][np.where(dists==dists.min())[0][0]].resid
    # better to just use CA instead of this atom for distance measurements since there's more variability in sidechain positions
    atom = u.atoms[A][np.where(dists==dists.min())[0][0]].name
    A_pos = u.select_atoms(f'chainID {seg_combo[0]} and resnum {resid} and name CA').positions[0]
    # center of mass of seg_combo[1]
    B_com = u.select_atoms(f'chainID {seg_combo[1]}').center_of_mass()
    # this identifies A's relationship to B. 
    comparison_dist = np.linalg.norm(A_pos-B_com)
    comparison_angle = np.rad2deg(get_angle(A_pos,com,B_com))
     
    for key, seg_list in identical_subunits.items():
        if seg_combo[0] in seg_list and seg_combo[1] in seg_list:
            A_group, B_group = key, key
        elif seg_combo[0] in seg_list and seg_combo[1] not in seg_list:
            A_group = key
        elif seg_combo[1] in seg_list and seg_combo[0] not in seg_list:
            B_group = key

    relationships = []
    for seg1 in identical_subunits[A_group]:
        distances = {}
        distance_difs = {}
        angle_difs = {}
        for seg2 in identical_subunits[B_group]:
            if seg1 != seg2:
            
                pos1 = u.select_atoms(f'chainID {seg1} and resnum {resid} and name CA').positions[0]
                pos2 = u.select_atoms(f'chainID {seg2}').center_of_mass()
                distance = np.linalg.norm(pos1-pos2)
                distances[(seg1,seg2)] = distance
                angle_difs[(seg1,seg2)] = np.abs(np.rad2deg(get_angle(pos1,com,pos2)) - comparison_angle)
                distance_difs[(seg1,seg2)] = np.abs(distance-comparison_dist)

        min_dist, min_angle = min(distance_difs, key=distance_difs.get), min(angle_difs, key=angle_difs.get)  
        if min_dist != min_angle:
            # sorting everything to ensure alphabetical order will align with contact naming scheme
            if ((distance_difs[min_dist]) + (angle_difs[min_dist])) < (distance_difs[min_angle]) + (angle_difs[min_angle]):
                relationships.append(tuple(sorted(min_dist)))
            else:
                relationships.append(tuple(sorted(min_angle)))
        else:       
            relationships.append(tuple(sorted(min_dist)))
    return relationships

def get_equivalent_interactions(representative_chains, u):
    '''
    For each chain in representative_chains, find all other identical chains' interaction partners
    that are equivalent to the representative_chains interactions with all the other chains in the protein.
    For instance, chain A's interaction with a subunit D on the other side of the protein might be equivalent to
    chain B's interaction with subunit E for a symmetric multimer.

    This is useful when averaging the contact frequencies of a multimer and determining the correct naming
    for the averaged contact record and to ensure it's depicted correctly when visualized.

    Parameters
    ----------
    representative_chains : list of strings
        A list with the names of the chain ids that you want to establish equivalent interactions for

    '''

    segids = u.segments.segids
    all_chain_dists = get_all_chain_dists(u)
    identical_subunits = find_identical_subunits(u)

    equivalent_interactions = {}
    for chain in [chain for chain in representative_chains]:
        for segid in segids:
            if segid != chain:
                equivalent_interactions[tuple(sorted((chain,segid)))] = asymmetric_measurements((chain,segid),identical_subunits,u, all_chain_dists)

    return equivalent_interactions