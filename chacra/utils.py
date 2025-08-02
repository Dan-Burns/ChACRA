import numpy as np
import re
import psutil
import parmed as pmd
import pandas as pd

def make_contact_frequency_dictionary(freq_files:list) -> pd.DataFrame:
    '''
    Deprecated in favor of make_contact_dataframe().
    go through a list of frequency files and record all of the frequencies for 
    each replica.  

    freq_files : list
        List of paths to each contact frequency file, presorted.

    Returns : Dict
        The Dictionary of contact keys and frequency lists.
    ''' 
    contact_dictionary = {}
  
    regex = r'\w:\w+:\d+\s+\w:\w+:\d+'
    # go through each of the contact files and fill in the lists
    for i, file in enumerate(freq_files):
        with open(file, 'r') as freqs:
            for line in freqs.readlines():
                if re.search(regex, line):
                    line = line.strip()
                    first, second, num_str = line.split()
                    label = first + "-" + second
                    
                    
                    if label not in contact_dictionary.keys():
                        contact_dictionary[label] = [0 for n in range(i)]
                        contact_dictionary[label].append(float(num_str))
                    else:
                        contact_dictionary[label].append(float(num_str))
        
        #Extend all the lists before opening the next freq_file
        for key in contact_dictionary.keys():
            if i > 0 and len(contact_dictionary[key]) != i+1:
                length = len(contact_dictionary[key])
                extend = (i+1) - length
                contact_dictionary[key].extend([0 for n in range(extend)])
            
    return contact_dictionary

def sort_dictionary_values(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: -item[1]))

def parse_id(contact):
    '''
    take the contact name (column id) and return a dictionary of
    the residue A descriptors and residue B descriptors
    '''
    chaina, resna, resida, chainb, resnb, residb = re.split(":|-", contact)
    ### for combined contact data, the prepended name needs to be removed from 
    ### chain a
    ##### This might break something if multiple contacts
    #### are going into the keys of another dictionary because
    #### duplicate names will be overwritten.
    ## shouldn't be a problem for averaging functions because combined data
    ## will be produced from pre-averaged data
    ## to_heatmap() will not give correct results as is - need to prepare
    ## the data with original names for that....
    
    if "_" in chaina:
        chaina = chaina.split("_")[1]

    return {'chaina':chaina, 'resna':resna, 'resida':resida,
             'chainb':chainb, 'resnb':resnb, 'residb':residb}

def split_id(contact):
    '''
    take the contact name and split it into its two residue parts
    returns a dictionary where 'resa' will contain 'CH:RES:NUM'
    '''
    resa, resb = re.split("-", contact)
    return {'resa':resa, 'resb':resb}
    
def multi_intersection(lists, cutoff=None, verbose=False):
    '''
    Return the intersection of the values in lists.  
    Parameters
    ----------
    lists : list of lists
        The lists of values to identify the shared elements from.

    cutoff : float or int
        If not None, return the intersection of a subset of lists that meet the criteria.
        float < 1 will only include lists that have a length of cutoff percent of
        the longest list.
        int > 1 will only include lists that are longer than cutoff.

    verbose : bool
        If True, print the number of lists that were provided as input and the number
        of lists that were used in constructing the intersection.

    Returns
    -------
    list
    intersection of values in lists.
        
    '''

    initial = len(lists)
    if cutoff is not None and cutoff < 1:
        longest_len = max([len(data) for data in lists])
        lists = [data for data in lists if len(data) > longest_len*cutoff]
    elif cutoff is not None and cutoff > 1:
        lists = [data for data in lists if len(data) > cutoff]

    final = len(lists)
    set1 = set(lists[0])
    setlist = [set(data) for data in lists[1:]]
    if verbose == True:
        print(f'n lists initial: {initial} \nn lists final: {final}')
    return sorted(list(set1.intersection(*setlist)))


def sort_nested_dict(d):
    '''
    Sort the split sum dictionary. This is expecting the keys of the nested dictionary to be
    in the form of "A:ALA:5". 
    '''
    sorted_dict = {}
    for outer_key, nested_dict in d.items():
        sorted_keys = sorted(nested_dict.keys(), key=lambda x: (x.split(":")[0], int(x.split(":")[-1])))
        sorted_nested_dict = {key: nested_dict[key] for key in sorted_keys}
        sorted_dict[outer_key] = sorted_nested_dict
    return sorted_dict

def get_resources():
    resources = {
    'num_cores': psutil.cpu_count(logical=False),  # physical cores
    'num_threads': psutil.cpu_count(logical=True),  # includes hyperthreads
    'total_ram_gb': psutil.virtual_memory().total / 1e9,  
    'available_ram_gb': psutil.virtual_memory().available / 1e9,
    'available_ram_mb': psutil.virtual_memory().available / 1e6  
    }
    return resources

def parmed_underscore_topology(gromacs_processed_top, atom_indices, output_top):
    '''
    Add underscores to atom types of selected atoms.
    This is useful if using the plumed_scaled_topologies script 
    for hremd system modification.
    With this, you still need to open the new topology file and delete the 
    underscores from the beginning of the file [atomtypes]
    or else plumed will look for atoms with 2 underscores to apply lambda to.
    '''
    top = pmd.gromacs.GromacsTopologyFile(gromacs_processed_top)

    for atom in top.view[atom_indices].atoms:
        atom.type = f"{atom.type}_"
        if atom.atom_type is not pmd.UnassignedAtomType:
            atom.atom_type = copy.deepcopy(atom.atom_type)
            atom.atom_type.name = f"{atom.atom_type.name}_"


    top.save(output_top)
