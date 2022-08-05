import json
import re
import pandas as pd

def count_contact_types(contact_type_dict):
    '''
    count the number of occurrences of each contact type per contact pair
    Takes the dictionary that contains contact pair keys and list of contact types values
    splits those lists into a new dictionary that has contact pair keys and dictionaries 
    of contact type keys and number of instances as values 
    then converts it to a dataframe
    '''
    
    nested_contact_types_per_contact_pair = {key:{'hbbb':0, 'hbsb':0, 'hbss':0, 
                 'lbs':0, 'lbb':0, 
                 'sb':0, 'pc':0, 'ps':0, 'ts':0, 'hp':0, 'vdw':0} for key in contact_type_dict.keys()}
    
    for contact in nested_contact_types_per_contact_pair.keys():
        for key in contact_type_dict.keys():
            if contact == key:
                for contact_type in nested_contact_types_per_contact_pair[contact].keys():
                    count = 0
                    for ele in contact_type_dict[key]:
                        if ele == contact_type:
                            count += 1
                    nested_contact_types_per_contact_pair[contact][contact_type] = count
                
    contact_types_df = pd.DataFrame.from_dict(nested_contact_types_per_contact_pair)
                
    return contact_types_df






















# Following are needed for parsing the original getcontacts file that contains all the contact information

def parse_contact(line):
    '''
    split the contact line into parts
    should probably use getcontacts line parsing scheme
    '''
    frame, contact_type, ch1, resn1, resid1, atom1, ch2, resn2, resid2, atom2, distance, _ = re.split("\s+|:", line)
    return frame, contact_type, ch1, resn1, resid1, atom1, ch2, resn2, resid2, atom2, distance
    
def make_contact_name(ch1,resn1,resid1,ch2,resn2,resid2):
    
    contact_name = ch1+":"+resn1+":"+resid1+"-"+ch2+":"+resn2+":"+resid2
    
    return contact_name
    
def tally_contact_type(contact_type, dictionary):
    '''
    keep track of the number of times a contact occurs with the contact_types dictionary
    '''
    return dictionary[contact_type] + 1
    

def parse_contact_2(line):
    '''
    This can identify correlated contacts and tell which part of which residue
    is making the contact
    '''
    line = line.rstrip()
    frame, contact_type, res1, res2, distance = re.split("\s+", line)
    return frame, contact_type, res1, res2
    
    
    
    
    

    
    
'''    
    
#The following is a template for a loop for getting all the contact types (run on hpc)
construct = 'teeic'

#open file containing the list of contacts to parse
with open('contact_types/'+construct+'_mutant_contact_ids_list.json') as d:
    contact_ids = json.load(d)

for i in range(1,20):
    
    
    contacts = construct+'_whole_cont_'+str(i)+'.tsv'
    
    f = open(contacts, 'r')

    f.seek(0)
    contact_type_dict = {'hbbb':0, 'hbsb':0, 'hbss':0,
                     'lbs':0, 'lbb':0,
                     'sb':0, 'pc':0, 'ps':0, 'ts':0, 'hp':0, 'vdw':0}

    subset_dictionary = make_subset_dictionary(contact_ids)

    for line in f.readlines()[2:]:

        frame, contact_type, ch1, resn1, resid1, atom1, ch2, resn2, resid2, atom2, distance = parse_contact(line)

        contact_name = make_contact_name(ch1,resn1,resid1,ch2,resn2,resid2)

        contact_type_dict[contact_type] = tally_contact_type(contact_type, contact_type_dict)

        if contact_name in subset_dictionary.keys():
            subset_dictionary[contact_name].append(contact_type)

    with open('contact_types/'+construct+str(i)+'_loop_mutant_types_dictionary.json', 'w') as g:
        json.dump(subset_dictionary, g)


    with open('contact_types/'+construct+str(i)+'_all_types_dictionary.json', 'w') as h:
        json.dump(contact_type_dict, h)
'''