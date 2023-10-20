

# break up average contacts into manageable functions
from itertools import combinations
import pandas as pd
import numpy as np
import re
from ChACRA.ContactAnalysis.contact_functions import _parse_id
from MDAnalysis.analysis.distances import distance_array

#TODO add option to globally define chacra color scheme
# this will be imported into the plot and pymol modules
chacra_colors = ['red','#02a8f8','#00b730','#7400ff','#434343','magenta','#fad300']
 
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
            identical_subunits[i] = list(subunits)

    return identical_subunits

def get_all_subunit_contacts(identical_subunits, df):
    '''
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
                # measuring angle between two subunits with the combined com as the vertex
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



#####
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


def get_contacting_chains():
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