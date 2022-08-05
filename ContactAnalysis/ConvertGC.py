# -*- coding: utf-8 -*-
"""
Functions to deal with the GetContacts files and turn them into dataframes.
should turn these into objects that will be converted to dataframes on the fly
so that you can deal with the huge amounts of information better.
e.g. look at all of the contacts involving just side chain atoms of one residue 
with all other contact partners.
"""
def make_contact_frequency_dictionary(freq_files):
    '''
    go through a list of frequency files and record all of the frequencies for 
    each replica
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

# Following are needed for parsing the original getcontacts file that contains all the contact information

def parse_contact(line):
    '''
    split the contact line into parts
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