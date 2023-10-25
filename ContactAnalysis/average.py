import MDAnalysis as mda
import numpy as np
import pandas as pd
from ChACRA.ContactAnalysis.ContactFrequencies import *
from ChACRA.ContactAnalysis.contact_functions import _parse_id
from ChACRA.ContactAnalysis.utils import *


def average_multimer(structure, denominator, df, representative_chains):
    '''
    This should be able to replace ContactFrequencies.average_contacts.

    Takes a contact df, pdb structure, and a selection of chains to use for representing the average contacts and finds all of the equivalent interactions
    and returns a dataframe of the averaged values.  These values are more robust and easier to visualize

    Some todos:
    #TODO figure out non redundant equivalent interactions and make that the denominator if the contact comes from that set
    #  denominator = len(longest_identical_chain_group participating in intersubunit contact)
    # TODO denominator needs to be cut in half if separate identical chains are making contact between the same resid and they're 180 deg to one another based off the entire complex com
    # something like this will help deal with opposing subunit contacts inside of a channel pore (might not really be needed)
    # ALSO - add a function to automatically identify the representative_chains based on identical_chain type, alphabetical order and proximities to one another
        

    select one of each type of chain to be the representative chain for average naming
    '''

    u = mda.Universe(structure)
    protein = u.select_atoms('protein')
    u = mda.Merge(protein)
    # For now can leave it equal to the minimum number of identical subunits involved in the contact 
    #(if it's between non-identical subunits, choose the subunit that has fewer identical ones)
    denominator = 6
    df_copy = df.copy()
    # hold the averaged data
    averaged_data = {}

    # determine what the equivalent chain interactions relative to representative chains are for all the subunits
    print("Finding interactions equivalent to those involving representative_chains. One moment.")
    equivalent_interactions = get_equivalent_interactions(representative_chains,u)
    
    print('Collecting equivalent contacts and averaging.')
    total_count = len(df.columns)
    with tqdm.tqdm(total=total_count) as progress:

        while len(df_copy.columns) > 0:
            resids = _parse_id(df_copy.columns[0])

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
                        contact_info = _parse_id(contact_name)
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
                        contact_info = _parse_id(contact_name)
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

                            # so we measure the distance between the contacting residues for each contact and determine the distance that these residues should be apart
                            # and then test the representative_pair residue distances with the default name and with the flipped resid info
                            ################# measure each distance in to_average's contacts ##################################
                            contact_distances = []
                            for contact_name in to_average:
                                contact_info = _parse_id(contact_name)
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
            
            averaged_data[averaged_name] = df_copy[to_average].sum(axis=1)/denominator

            # get rid of the contacts that were just averaged and reduce the number of remaining contacts to check in the dataframe
            df_copy.drop(to_average, axis=1, inplace=True)
            # update the progress bar
            columns_removed = len(to_average)
            progress.update(columns_removed)

    return pd.DataFrame(averaged_data)