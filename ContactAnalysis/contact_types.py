import json
import re
import pandas as pd
from statistics import mode


#TODO return directionality of contact i.e. when one is contributing the sidechain then the direction goes from that one 

# loop to tabulate per frame contacts is in old/older/sandbox_trp 
# uses from getcontacts.contact_calc.transformations import *
#https://getcontacts.github.io/interactions.html


########## res_contacts taken directly from getcontacts to avoid dependencies ##################
########## Easy way of recording the per frame contact pairs ####################
########## Can modify to incorporate the edge directionality ####################
def res_contacts(contacts):
    """
    Convert atomic contacts into unique residue contacts. The interaction type is removed as well as any third or
    fourth atoms that are part of the interaction (e.g. water-bridges). Finally, the order of the residues within an
    interaction is such that the first is lexicographically smaller than the second ('A:ARG' comes before 'A:CYS').

    Example
    -------
        res_frequencies([
            [0, 'hbbb', 'A:ASN:108:O', 'A:ARG:110:N'],
            [0, 'vdw', 'A:ARG:110:N', 'A:ASN:108:CB']
        ])
        # => [[0, 'A:ARG:110', 'A:ASN:108']]

    Parameters
    ----------
    contacts: List of list
        List of contacts, where each contact is a list of frame-num, i-type, and atom names

    Returns
    -------
    List of list
        Each entry is a list with a frame and two residue identifiers
    """
    # print(contacts)
    from collections import defaultdict
    # Associates a frame-number with a set of contacts
    frame_dict = defaultdict(set)

    for atom_contact in contacts:
        frame = atom_contact[0]
        resi1 = ":".join(atom_contact[2].split(":")[0:3])
        resi2 = ":".join(atom_contact[3].split(":")[0:3])
        if resi2 < resi1:
            resi1, resi2 = resi2, resi1
        frame_dict[frame].add((resi1, resi2))

    ret = []
    for frame in sorted(frame_dict):
        for resi1, resi2 in frame_dict[frame]:
            ret.append([frame, resi1, resi2])

    return ret




def res_contacts_xl(input_lines, itypes=None):
    """
    This was for the getcontacts branch
    
    Reduces memory usage for contact frequency calculation by combining parse_contact and res_contacts.
    Read a contact-file (tab-separated file with columns: frame, i-type, atomid1, atomid2[, atomid3[, atomid4]] 
    one frame at a time and run res_contacts after each frame.  
    The total number of frames is also returned.

    Example
    -------
        parse_contacts([
            "# total_frames:2\n",
            "0  hbbb    A:ALA:1:N   A:THR:10:O\n",
            "0  vdw     A:ALA:1:CB  B:CYS:3:H\n",
            "1  vdw     A:ALA:1:N   A:THR:10:C\n"
        ])
        # returns:
        # ([
        #        [0, "hbbb", "A:ALA:1:N", "A:THR:10:O"],
        #        [0, "vdw", "A:ALA:1:CB", "B:CYS:3:H"],
        #        [1, "vdw", "A:ALA:1:N", "A:THR:10:C"]
        #  ], 2)

    Parameters
    ----------
    input_lines: iterable
        Iterator of over a set of strings. Can be a file-handle

    itypes: set of str | None
        Interactions to include in the output

    Returns
    -------
    List of list
        Each entry is a list with a frame and two residue identifiers

    Raises
    ------
    ParseError: If contents of lines couldn't be parsed
    """
    ret = []
    # hold the contacts for a single frame
    frame_contacts = []
    total_frames = 0
    # track the current frame
    current_frame = 0

    for line in input_lines:
        # check for end of file 
        if line == '':
            ret.extend(res_contacts(frame_contacts))
        line = line.strip()
        if "total_frames" in line:
            tokens = line.split(" ")
            total_frames = int(tokens[1][tokens[1].find(":")+1:])

        if len(line) == 0 or line[0] == "#":
            continue

        tokens = line.split("\t")
        try:
            tokens[0] = int(tokens[0])
        except ValueError:
            raise ParseError("First column isn't a integer")
        
        if len(tokens) not in range(4, 8):
            raise ParseError("Invalid number of tokens")
        # check that the data is from the same frame
        if tokens[0] == current_frame:
            if itypes is None or tokens[1] in itypes:
                frame_contacts.append(tokens)
        # if it's not, convert the previous frame to single contact records
        # update the frame record
        # start a new frame_contacts list
        # check for end of file
        else:
            ret.extend(res_contacts(frame_contacts))
            frame_contacts = []
            current_frame = tokens[0]


    return ret, total_frames
######################################################################################

## IN PROGRESS ##
'''
with open(per_frame_data,'rb') as f:
    # return the first 100 MB
    data = f.readlines(100000000) 



'''

def get_contact_directions(data):
    '''
    Get the directional contact frequency where a residue in a contact pair
    gets weight if its sidechain is contributing to the contact.

    A separate record is maintained for backbone-backbone contacts.

    This data can be used for directed graph analyses. 
    '''
    records = {}
    backbone = {'N','CA','C','O'}
    n_frames = int(data[0].split()[1].split(':')[1])
    for line in data:
        info = re.split(r'\s+',line)
        # format: ['0', 'sb', 'A:GLU:185:OE1', 'A:LYS:184:NZ', '2.753']
        if info[0] == '#':
            continue
        
        #if info[1] not in ['hp', 'hbsb', 'vdw']:
        rec1 = info[2].split(":")
        rec2 = info[3].split(":")
        resi1 = ":".join(rec1[0:3])
        resi2 = ":".join(rec2[0:3])
        contact = f'{resi1}-{resi2}'
        if contact not in records.keys():
            records[contact] = {resi1:0, resi2:0, 'bb':0}
        if rec1[3] not in backbone:
            records[contact][resi1]+=(1/n_frames)
        if rec2[3] not in backbone:
            records[contact][resi2]+=(1/n_frames)
        if rec1[3] in backbone and rec2[3] in backbone:
            records[contact]['bb']+=(1/n_frames)
      
    return records
    # determine who the side chain donor is
    # backbone atoms are CA, C, O, N

backbone_atoms = {'N','CA','C','O'}
# Right now this is just concerned with hbsb
# hbss, hbbb, pc, ps, ts, sb can be inferred from the type
# and are all treated as bidirectional edges
# hp, hbsb, vdw need to be checked 
# hp and vdw could be bidirectional
# need to add weights such that if you have a mix of side chain and backbone
# interactions between 2 residues, the one that donates the sc more has higher weight.
'''
dictionary = {contact_name: {resa:n sidechain donations,
                                 resb:n side chain dontations,
                                 bb: n backbone-backbone contacts
                            }
            }
 ''' 
def get_sc_donor(line_list):
    '''
    Take a line from per_frame_data and determine which
    residue is contributing the side chain

    Returns
    -------
        single element list of resname contributing side chain
    '''
    res1 = line_list[2]
    res2 = line_list[3]
    donor = [res for res in [res1, res2] if res.split(":")[3] not in backbone_atoms]
    
    return donor

def record_sc_donors(data):
    '''
    Alternative approach to get_contact_directions
    Returns
    -------
        dictionary of contact pair keys and sc donor values
        the sc donor is the most frequently occuring residue in the sc donor lists
    '''
    donors = {}
    for line in data:
        info = re.split(r'\s+',line)
        if info[1] == 'hbsb':
            resi1 = ":".join(info[2].split(":")[0:3])
            resi2 = ":".join(info[3].split(":")[0:3])
            if resi2 < resi1:
                resi1, resi2 = resi2, resi1
            pair = f'{resi1}-{resi2}'
            try:
                donors[pair].append(":".join(get_sc_donor(info)[0].split(":")[0:3]))
            except:
                donors[pair]=[":".join(get_sc_donor(info)[0].split(":")[0:3])]
    # This is not going to catch sidechain-sidechain interactions   
    # instead of mode, get the actual counts from each list/n_frames
    return {name:mode(donors[name]) for name in donors}


def directed_contact_edges(donor_edges):
    '''
    Returns
    -------
        edge list in digraph format
    '''
    # networkx wants directed edges in the format of 
    # G = nx.DiGraph()
    # G.add_edges_from([(1, 2), (1, 3)])
    # Then go back through and add weights using cont_obj.all_edges 
    directed_edges = []
    for pair, donor in donor_edges.items():
        resi1, resi2 = pair.split("-")
        directed_edges.append((donor,[res for res in [resi1,resi2] if res != donor][0]))
    return directed_edges

## END IN PROGRESS ##


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
    '''
    
    '''
    
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