import MDAnalysis as mda
import numpy as np
import pandas as pd
from .ContactFrequencies import *
from .utils import *
import tqdm


def average_multimer(structure, denominator, df, representative_chains):
    '''
    This should be able to replace ContactFrequencies.average_contacts.

    Takes a contact df, pdb structure, and a selection of chains to use for representing the average contacts and finds all of the equivalent interactions
    and returns a dataframe of the averaged values.  These values are more robust and easier to visualize

    Parameters
    ----------
    structure : str
        Path to pdb file that contact frequencies/ simulations are based on.

    denominator : int
        The number of subunits to devide the contact frequency sums by.  Planning to make this automated or accept lists for unique situations.
        e.g. If you have a tetramer, denominator is 4.  If you have a heteromultimer with 6 subunits of each type, denominator is 6.

    df : pd.DataFrame
        The contact frequency dataframe to average.
    
    representative_chains : list
        List of pdb chain ids to use for depicting the averaged contacts on.
        e.g. if you have a tetramer of 'A', 'B', 'C', 'D' subunits, representative_chains=['A'].
        If you have a heteromultimer, select one of each chain type that are in contact with one another. representative_chains=['A','G'].

    Returns
    -------
    pd.DataFrame containing the values for averaged among the identical contacts.

    Some todos:
    #TODO figure out non redundant equivalent interactions and make that the denominator if the contact comes from that set
    #  denominator = len(longest_identical_chain_group participating in intersubunit contact)
    # TODO denominator needs to be cut in half if separate identical chains are making contact between the same resid and they're 180 deg to one another based off the entire complex com
    # something like this will help deal with opposing subunit contacts inside of a channel pore (might not really be needed)
    # ALSO - add a function to automatically identify the representative_chains based on identical_chain type, alphabetical order and proximities to one another
    # opposing chain iding and handling.
    
    '''

    u = mda.Universe(structure)
    protein = u.select_atoms('protein')
    u = mda.Merge(protein)
    # For now can leave it equal to the minimum number of identical subunits involved in the contact 
    #(if it's between non-identical subunits, choose the subunit that has fewer identical ones)
    identical_subunits = find_identical_subunits(u)
    df_copy = df.copy()
    # hold the averaged data
    averaged_data = {}
    # standard_deviation = {} # use this to compute error later
    # determine what the equivalent chain interactions relative to representative chains are for all the subunits
    print("Finding interactions equivalent to those involving representative_chains. One moment.")
    equivalent_interactions = get_equivalent_interactions(representative_chains,u)
    
    print('Collecting equivalent contacts and averaging.')
    total_count = len(df.columns)
    with tqdm.tqdm(total=total_count) as progress:

        while len(df_copy.columns) > 0:
            resids = parse_id(df_copy.columns[0])

            # find all of the other contacts that involve these residue names and numbers
            regex = make_equivalent_contact_regex(resids)
            to_average = list(df_copy.filter(regex=regex, axis=1).columns)
            # chaina and chainb are picked up from this iterations initial contact 
            chaina, chainb = resids['chaina'], resids['chainb']
            if chaina == chainb:
                identical_pair = True
            else:
                identical_pair = False
            # get the chain id pair that the averaged name will be based on
            # can't determine the averaged name yet because inter-subunit contacts can have flipped resids (A-> B can be equivalent to C->A but this is recorded as A->C 
            # but the resids are swapped)
            representative_pair = get_representative_pair_name(chaina, chainb, identical_subunits, representative_chains, equivalent_interactions)

            ######################################### Determine Average Contact Name and Filter out things that shouldn't be in to_average ###########
            ####################################### Happening in same subunit ##############################################################
            if identical_pair:
                # The easiest (and most common) case
                averaged_name =  f"{representative_pair[0]}:{resids['resna']}:{resids['resida']}-{representative_pair[1]}:{resids['resnb']}:{resids['residb']}"
                to_drop = []
                for contact_name in to_average:
                        contact_info = parse_id(contact_name)
                        # if the chaina variable isn't in the same identical subunit group as the current contact's chain you know it's wrong
                        # drop it (this means it happens to have the same resname and resnum but is happening on a different kind of subunit - very unlikely)
                        if (contact_info['chaina'] not in identical_subunits[get_chain_group(chaina, identical_subunits)]):
                            to_drop.append(contact_name)
                for contact_name in to_drop:
                        to_average.remove(contact_name)
            ##################################### Happening Inter-Subunit ########################################################
            else:
                # filter and catch the averaged_name if it's in to_average
                to_drop = []
                matched_name = None
                
                for contact_name in to_average:
                        contact_info = parse_id(contact_name)
                        # drop everything that's not from the equivalent_interaction set
                        if (contact_info['chaina'],contact_info['chainb']) not in equivalent_interactions[representative_pair]:
                            to_drop.append(contact_name)
                        # if one of to_average involves both of representative_pair's chains, then the name is determined    
                        if (contact_info['chaina'], contact_info['chainb']) == representative_pair:
                            matched_name = contact_name

                # drop the ones that don't fit the original pair
                for contact_name in to_drop:
                        to_average.remove(contact_name)


                # if inter-hetero subunit     
                if get_chain_group(representative_pair[0], identical_subunits) != get_chain_group(representative_pair[1], identical_subunits):
                        # and the order that the hetero subunit appears in representative_pairs matches, you can name it without further consideration 
                        if get_chain_group(representative_pair[0], identical_subunits) ==  get_chain_group(resids['chaina'], identical_subunits):
                            averaged_name =  f"{representative_pair[0]}:{resids['resna']}:{resids['resida']}-{representative_pair[1]}:{resids['resnb']}:{resids['residb']}"
                else: 
                        # if a contact was found earlier that matches the representative pair, you name it now
                        if matched_name is not None:
                            averaged_name = matched_name

                        else:
                            # have to determine if the original contact for this iteration happens to have the flipped inter-subunit naming
                            # and to_average didn't include the contact with the representative naming scheme (should not happen often)
                            # TODO can't flip names for inter-hetero subunit....
                            # so we measure the distance between the contacting residues for each contact and determine the distance that these residues should be apart
                            # and then test the representative_pair residue distances with the default name and with the flipped resid info
                            ################# measure each distance in to_average's contacts ##################################
                            contact_distances = []
                            for contact_name in to_average:
                                contact_info = parse_id(contact_name)
                                sel1 = f"chainID {contact_info['chaina']} and resnum {contact_info['resida']} and name CA"
                                sel2 = f"chainID {contact_info['chainb']} and resnum {contact_info['residb']} and name CA"
                                contact_distances.append(get_pair_distance(sel1, sel2, u))

                            #### identify the correct contact averaged_name by checking the two possible distances ########################
                            mean_distance = np.mean(contact_distances)

                            testa1 = f"chainID {representative_pair[0]} and resnum {contact_info['resida']} and name CA"
                            testa2 = f"chainID {representative_pair[1]} and resnum {contact_info['residb']} and name CA"
                            testb1 = f"chainID {representative_pair[1]} and resnum {contact_info['resida']} and name CA"
                            testb2 = f"chainID {representative_pair[0]} and resnum {contact_info['residb']} and name CA"

                            testa = get_pair_distance(testa1, testa2, u)
                            testb = get_pair_distance(testb1, testb2, u)
                            
                            # if the difference between the mean contact distance distance and the chain swapped one is greater....
                            if np.abs(testa-mean_distance) < np.abs(testb-mean_distance):
                                averaged_name = f"{representative_pair[0]}:{resids['resna']}:{resids['resida']}-{representative_pair[1]}:{resids['resnb']}:{resids['residb']}"
                            else:
                                averaged_name =  f"{representative_pair[0]}:{resids['resnb']}:{resids['residb']}-{representative_pair[1]}:{resids['resna']}:{resids['resida']}"
                            ##################################### End flipped contact name adjustment ##########################################################################
            ################################ End inter-subunit #####################################################################################
            # TODO record stdev of every averaged contact 
            averaged_data[averaged_name] = df_copy[to_average].sum(axis=1)/denominator

            # get rid of the contacts that were just averaged and reduce the number of remaining contacts to check in the dataframe
            df_copy.drop(to_average, axis=1, inplace=True)
            # update the progress bar
            columns_removed = len(to_average)
            progress.update(columns_removed)

    return pd.DataFrame(averaged_data)


def everything_from_averaged(averaged_contacts, original_contacts, u, representative_chains,
                             as_map=False):
    '''
    Take the averaged contacts and regenerate the entire protein's contacts using this data.
    Useful for visualizing the flow of chacras across the whole protein and doing network analysis
    on more statistically robust data.

    averaged_contacts : pd.DataFrame
        The averaged contact data

    u : MDA.Universe 
        The same universe used for averaging contacts

    representative_chains : list of strings
        The list of chain ids used when generating the averaged contact names.

    as_map : bool
        Returns a dictionary mapping each averaged contact name to a list of the corresponding
        replicated contact names

    TODO Reduce the size of this function - one call to append to replicated contacts per iteration
    TODO function to generate alternate names and measure distances - can be used in average_hetermultimer too

    Returns
    -------
    pd.DataFrame of the averaged contact values applied back to all chains.
    '''
    print("Collecting some information. One moment.")
    protein = u.select_atoms('protein')
    u = mda.Merge(protein)
    # For now can leave it equal to the minimum number of identical subunits involved in the contact 
    #(if it's between non-identical subunits, choose the subunit that has fewer identical ones)
    equivalent_interactions = get_equivalent_interactions(representative_chains,u)
    replicated_contacts = {}
    mapped_contacts = {contact:[] for contact in averaged_contacts.columns}
    #unreplicated_contacts = []
    identical_subunits = find_identical_subunits(u)
    for contact in tqdm.tqdm(averaged_contacts.columns):
        resids = parse_id(contact)
        chaina = resids['chaina']
        chainb = resids['chainb']
        if chaina == chainb:
             for identical_subunit in identical_subunits[get_chain_group(chaina,identical_subunits)]:
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
                # now have to check every time to determine if the averaged_contact happens to have
                # a flipped name relative to the other chain pairs
                testa = f"{equivalent_pair[0]}:{resids['resna']}:{resids['resida']}-{equivalent_pair[1]}:{resids['resnb']}:{resids['residb']}"
                testb = f"{equivalent_pair[0]}:{resids['resnb']}:{resids['residb']}-{equivalent_pair[1]}:{resids['resna']}:{resids['resida']}"
                testa_in = testa in original_contacts.columns
                testb_in = testb in original_contacts.columns
                # if both contact names are in the original data, then measure distances and compare them to the averaged contact
                # if they're hetero-subunit, can name without further consideration
                if get_chain_group(chaina, identical_subunits) != get_chain_group(chainb, identical_subunits):
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
                    # can print pairs of distances that are under some difference threshhold
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
                    # can print pairs of distances that are under some difference threshhold
                    if np.abs(testa_dist-ref_dist) < np.abs(testb_dist-ref_dist):
                        equivalent_contact = testa
                    else:
                        equivalent_contact = testb
                    replicated_contacts[equivalent_contact] = averaged_contacts[contact]
            
                mapped_contacts[contact].append(equivalent_contact)
                #unreplicated_contacts.append(equivalent_contact)
                    
    if as_map:
         return mapped_contacts
    else:
        return replicated_contacts #, unreplicated_contacts

#### For depicting selections of high loading score contacts with pymol functions
## need to provide the averaged dataframe with just the high loading score contacts
## Then retrieve the data for writing the pymol selection for that contact
## then find the equivalent contacts to that one and write the same pymol selection
## with just the new resid info replacing the averaged name data
# use get_contact_data on the averaged data, then duplicate the entries for all of the replicated contacts


############## Averaging Utilities #################################


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

def get_chain_group(chain, identical_subunits):
    '''
    Return the identical subunit group that the subunit is in
    '''
    for group in identical_subunits:
        if chain in identical_subunits[group]:
            return group
        
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



def make_equivalent_contact_regex(resids):
     '''
     resids : the parse_id dictionary containing the contact data
     #TODO can remove any potential ambiguity by adding the list of correct chain group chains 
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