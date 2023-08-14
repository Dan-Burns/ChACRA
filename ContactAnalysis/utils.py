

# break up average contacts into manageable functions
from itertools import combinations
import pandas as pd
import numpy as np
from contact_functions import _parse_id

def find_identical_subunits(universe):
    '''
    determines which subunits are identical

    Parameters
    ----------
    universe: 
        mda.Universe() generated with the structure that the contact
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
    for combo in combinations([segid for segid in segids],2):
        bool = np.all(np.equal(residues[combo[0]],residues[combo[1]]))
        # Enter True or False in both the upper tri and lower tri
        identical_table[combo[1]][combo[0]], identical_table[combo[0]][combo[1]] =  bool, bool
    
    # Only keep one representative row that describes the identical sets of subunits
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
            print(f'Got more contacts than identical subunits for contact matching {contact}: {to_average}. Maybe this is in a channel pore?')
            return df[to_average].mean(axis=1)