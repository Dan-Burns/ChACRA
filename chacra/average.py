import MDAnalysis as mda
import numpy as np
import pandas as pd
from .ContactFrequencies import *
from .utils import *
import tqdm
import os


def average_multimer(structure, df, representative_chains, denominator=None,
                     return_stdev=False, 
                     check_folder='equivalent_interactions'):
    '''
    
    Takes a contact df, pdb structure, and a selection of chains to use for 
    representing the averaged contacts and finds all of the equivalent interactions
    and returns a dataframe of the averaged values.  These values are more robust
    and easier to visualize. Currently, this should only be used in cases where
    the subunits in the structure are all identical or there are equal numbers of 
    each unique subunit type.

    Parameters
    ----------
    structure : str
        Path to pdb file that contact frequencies/ simulations are based on.

    df : pd.DataFrame
        The contact frequency dataframe to average.
    
    representative_chains : list
        List of pdb chain ids to use for depicting the averaged contacts on.
        e.g. if you have a tetramer of 'A', 'B', 'C', 'D' subunits, 
        representative_chains=['A']. If you have a heteromultimer, select one of
        each chain type that are in contact with one another and will give you 
        the appropriate visualization. e.g. representative_chains=['A','G'].

    denominator : int
        The number of subunits to devide the contact frequency sums by.  If None
        (recommended), this value is determined automatically assuming there are
        only identical subunits in the structure or equal numbers of the unique
        subunit types. This implementation will likely change in a future version.
    
    return_stdev : bool
        If True, returns averaged_contacts and standard_deviation dataframes.
        Usage : avg_df, stdev = average_multimer(args)

    check_folder : str
        Name of folder to write verification files to. These should be used with
        pymol to confirm that the averaging is being done between the correct 
        pairs of subunits.
   
    Returns
    -------
    pd.DataFrame containing the values for averaged among the identical contacts.

    Some todos:
   
    
    # TODO situations with different numbers of different subunit types
    # need special handling.  As is, averaging should only be done with 
    # structures involving all identical subunits or equal numbers of each type
    # of subunits in a heteromultimer case. In ion channels there are also a 
    # handful of contacts that will make contact with the opposing subunit
    # i.e. only have two partners instead of 4, or even make contacts with 
    # identical partners on the opposing subunit and the neighboring subunit.
    
    '''

    u = mda.Universe(structure)
    protein = u.select_atoms('protein')
    ###### Only calculating contacts within the protein
    u = mda.Merge(protein)

    identical_subunits = find_identical_subunits(u)

    ##################### determine averaging denominator.  ##########
    if denominator is None:
        n_subunits_each = np.asarray([len(list(i)) for i in identical_subunits.values()])
        # If the subunits are not all identical and there are more of one type
        # of subunit than another, user must specify what to divide by for 
        # averaging.
        if np.all(n_subunits_each == n_subunits_each[0]):
            denominator = n_subunits_each[0]
        else:
            print("The number of non-identical subunits is not equal between types."\
                " \n Specify the value to dividy by for averaging with the" \
                " 'denominator' argument. \n")
    print(f'Sums of identical contacts will be divided by {denominator} '\
          'for averaging.')
    ############################################################################

    # removing the columns in placed, so using a copy of the original data
    df_copy = df.copy()
    # hold the averaged data
    averaged_data = {}
     # collect this for error bars on averaged contact vs temp plots
    standard_deviation = {}

    ########## determine what the equivalent chain interactions relative to 
    ############ representative chains are for all the subunits
    print("Finding interactions equivalent to those involving representative_chains. One moment. \n")
    equivalent_interactions = get_equivalent_interactions(representative_chains,u)
    if check_folder is not None:
        equivalent_interactions_check(equivalent_interactions, 
                                  output_folder=check_folder)
        print(f"Pymol selection files are saved in {check_folder} folder.\n"\
              "Confirm that averaging is being done between the correct subunit pairs that "\
              "are equivalent to those in the filename. \n"\
              "If there are issues, please post them to https://github.com/Dan-Burns/ChACRA/issues"
              )
    ########################################################################
    
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
                identical_pair = True
            else:
                identical_pair = False

            # then this chain pair is mapped back to the chains involving
            # the representative chains
            representative_pair = get_representative_pair_name(chaina, chainb, 
                                                               identical_subunits, 
                                                               representative_chains, 
                                                               equivalent_interactions)

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
                to_drop = []
                for contact_name in to_average:
                        contact_info = parse_id(contact_name)
                        # if the chaina variable isn't in the same identical subunit group as the current contact's chain you know it's wrong
                        # drop it (this means it happens to have the same resname and resnum but is happening on a different kind of subunit - very unlikely)
                        if (contact_info['chaina'] not in identical_subunits[get_chain_group(chaina, identical_subunits)]):
                            to_drop.append(contact_name)
                for contact_name in to_drop:
                        to_average.remove(contact_name)
            ################## Happening Inter-Subunit ######################
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


                ########################## if inter-hetero subunit ##########    
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
                            # If it hasn't been figured out at this point,
                            # we measure the distance between the contacting residues for each contact and determine the distance that these residues should be apart
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
            ### At this point, contacts incorrectly captured by regex filter
            ### should be dropped, and the correct averaged name determined.
            
            averaged_data[averaged_name] = df_copy[to_average].sum(axis=1)/denominator
            standard_deviation[averaged_name] = df_copy[to_average].std(axis=1)
            # get rid of the contacts that were just averaged and reduce the number of remaining contacts to check in the dataframe
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
   
    equivalent_interactions = get_equivalent_interactions(representative_chains,u)
    replicated_contacts = {}
    mapped_contacts = {contact:[] for contact in averaged_contacts.columns}
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
                # a flipped name relative to the other chain pairs (alphabetical order swapping)
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
 
                    
    if as_map:
         return mapped_contacts
    else:
        return replicated_contacts

#### For depicting the high loading score contacts on all subunits,
## return as_map, and provide the original averaged high loading score contacts
# and the map to pymol_averaged_chacras_to_all_subunits
#


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
     #TODO can remove any potential ambiguity by adding the list of correct chain 
     group chains 
     '''
     regex1 = f"[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)-[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)"
     regex2 = f"[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)-[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)"
     return f"{regex1}|{regex2}"

def get_representative_pair_name(chaina, chainb, identical_subunits, 
                                 representative_chains, equivalent_interactions):
     '''
     provide the chains from the contact and get the chain names to use for 
     making a generalized/averaged contact name
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
    Find the residue that creates the greates difference in distances between 
    other chains by finding a point in the chain that's near some other 
    neighboring chain.

    #TODO can also sample random points on the chain and take the one
     with the best differences
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
    This will identify all of the interaction pairs equivalent to seg_combo
    all_chain_dists is included so the best residue is used to create the best
      variance in the measured distances 
    
    seg_combo : tuple(string, string)
        The pair of chains that you want to identify all the equivalent 
        interacting pairs for. ('A','B') will find other identical chains pairs 
        that interact with the same geometry as A and B
    
    identical_subunits : dictionary
        The dictionary with integer values and lists of identical subunits.

    all_chain_dists : dictionary
        The dictionary with chain_id values and nest dictionaries giving the 
        chain keys and the minimum distance to them as values
    
    Returns
    -------
    list of tuples
    each tuple will be a pair of chains with equivalent interaction geometry
    
    '''

    # whole protein center of mass
    com = u.select_atoms('protein').center_of_mass()
    # indices of atoms corresponding to chain seg_combo[0]
    A = np.where(u.atoms.segids == f'{seg_combo[0]}')[0]
    # asymmetric center of mass that is some point on the periphery of 
    # seg_combo[0] near seg_combo[1]
    # if they're not within 5 angstroms, this won't work - need to just identify
    # closest chain, pick one at random and then establish the as_com with that
    neighbor = find_best_asymmetric_point(u, seg_combo[0], all_chain_dists)
    as_com = u.select_atoms(f'(around 5 chainID {neighbor}) and chainID ' \
                            f'{seg_combo[0]}').center_of_mass()
    # the distances between all the atoms in seg_combo[0] and the asymmetric 
    # center of mass
    dists = np.apply_along_axis(np.linalg.norm, 1,u.atoms[A].positions - as_com)
    #find the closest res to the as_com so that this residue can be used for the
    # reference point for other identical chains
    resid = u.atoms[A][np.where(dists==dists.min())[0][0]].resid
    # better to just use CA instead of this atom for distance measurements since 
    #there's more variability in sidechain positions
    atom = u.atoms[A][np.where(dists==dists.min())[0][0]].name
    A_pos = u.select_atoms(f'chainID {seg_combo[0]} and resnum {resid} and name CA'
                           ).positions[0]
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
            
                pos1 = u.select_atoms(f'chainID {seg1} and resnum {resid} and name CA'
                                      ).positions[0]
                pos2 = u.select_atoms(f'chainID {seg2}').center_of_mass()
                distance = np.linalg.norm(pos1-pos2)
                distances[(seg1,seg2)] = distance
                angle_difs[(seg1,seg2)] = np.abs(np.rad2deg(
                    get_angle(pos1,com,pos2)) - comparison_angle)
                distance_difs[(seg1,seg2)] = np.abs(distance-comparison_dist)

        min_dist, min_angle = min(distance_difs, key=distance_difs.get), min(
                                                angle_difs, key=angle_difs.get)  
        if min_dist != min_angle:
            # sorting everything to ensure alphabetical order will align with 
            #contact naming scheme
            if ((distance_difs[min_dist]) + (angle_difs[min_dist])
                ) < (distance_difs[min_angle]) + (angle_difs[min_angle]):
                relationships.append(tuple(sorted(min_dist)))
            else:
                relationships.append(tuple(sorted(min_angle)))
        else:       
            relationships.append(tuple(sorted(min_dist)))
    return relationships

def get_equivalent_interactions(representative_chains, u):
    '''
    For each chain in representative_chains, find all other identical chains' 
    interaction partners that are equivalent to the representative_chains 
    interactions with all the other chains in the protein. For instance, chain 
    A's interaction with a subunit D on the other side of the protein might be
    equivalent to chain B's interaction with subunit E for a symmetric multimer.

    This is useful when averaging the contact frequencies of a multimer and 
    determining the correct naming for the averaged contact record and to 
    ensure it's depicted correctly when visualized.

    Parameters
    ----------
    representative_chains : list of strings
        A list with the names of the chain ids that you want to establish 
        equivalent interactions for

    '''

    segids = u.segments.segids
    all_chain_dists = get_all_chain_dists(u)
    identical_subunits = find_identical_subunits(u)
    # deals with homodimer
    if len(identical_subunits) == 1 and len(segids) == 2:
        equivalent_interactions = {tuple(sorted((segids[0],segids[1])))
                                 : [tuple(sorted((segids[0],segids[1])))]
                                    }
    else:
        equivalent_interactions = {}
        for chain in [chain for chain in representative_chains]:
            for segid in segids:
                if segid != chain:
                    equivalent_interactions[tuple(sorted((chain,segid)))
                                            ] = asymmetric_measurements(
                                            (chain,segid),identical_subunits,
                                            u, all_chain_dists)

    return equivalent_interactions

def equivalent_interactions_check(equivalent_interactions, 
                                  output_folder='equivalent_interactions'):
    '''
    Confirm that the averaging is taking place between the appropriate subunits
    with a series of pymol selections.

    The pair names are in alphabetical order so the ordering of the letter
    does not necessarily correspond to the interaction order of the original 
    pair that each file is named after.
    '''
    if os.path.exists(output_folder):
        pass
    else:
        os.makedirs(output_folder)

    for pair, equivalent_pair_list in equivalent_interactions.items():
        with open(f'{output_folder}/chains_{pair[0]}-{pair[1]}_check.pml','w') as f:
            for equiv_pair in equivalent_pair_list:
                selection_string = f"select {equiv_pair[0]}-{equiv_pair[1]}, "\
                        f"chain {equiv_pair[0]} or chain {equiv_pair[1]}"
                f.write(f'{selection_string} \n')

