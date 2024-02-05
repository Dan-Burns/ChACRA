import re
import numpy as np
from scipy.stats import linregress 
import matplotlib as mpl
from pylab import cm
from matplotlib.colors import to_hex
import collections
from scipy.interpolate import interp1d
from ..utils import parse_id
from .colors import chacra_colors

# TODO offer red, blue etc spectrums to link individual contact plots to color codes on pymol depiction
# TODO update all the functions to just accept one contact object and add a single function to write a pymol file for specific chacra(s)
def get_contact_data(contact_list, contactFrequencies, contactPCA,
                    slope_range=(0,7),
                    pc_range=(1,4),
                    variable_sphere_transparency=False,
                    max_transparency=.9,
                    ):
    '''
    collect all the relevant data that the other functions will need to 
    sort, color, and draw lines.

    Parameters
    ----------
    contact_list : list
        List of the contacts in format chain1:resname1:resnum1-chain1:resname1:resnum1
    
    contactFrequencies : ContactFrequencies object
        from TSenCA.ContactAnalyis.ContactFrequencies
        object's methods provide easy access to relevant data

    contactPCA : contactPCA object
        from TSenCA.ContactAnalyis.ContactFrequencies

    slope_range : tuple of int
        the lowest and highest (row) indices (inclusive) from the contact data to calculate
        the slope of the contact frequencies within

    pc_range : tuple of int
        the lowest and highest PCs (inclusive and 1 indexed) to consider when
        identifying the highest loading scores, coloring, etc.
    
    Returns
    -------
    dictionary of dictionaries containing contact name and corresponding data for writing the pymol selections
    

    # TODO have to figure out how to deal with duplicate names in different pc groups so you can depict just spheres of one color 
    # TODO update to new get_scores function, eliminate 'rank' and anything but highest score
    '''
    data = {contact:{} for contact in contact_list}

    #easier access to contact dataframe
    cdf = contactFrequencies.freqs

    for contact in contact_list:
        chaina, resna, resia, chainb, resnb, resib = re.split(
                                                        ":|-", contact
                                                            )
        data[contact]['chaina'] = chaina
        data[contact]['resna'] = resna
        data[contact]['resia'] = resia
        data[contact]['chainb'] = chainb
        data[contact]['resnb'] = resnb
        data[contact]['resib'] =resib

        # get the PC that the contact scores highest on
        # dictionary where first key is top PC
        score = contactPCA.get_top_score(contact, pc_range=(pc_range[0],pc_range[1]))

        top_pc = list(score.keys())[0]

        top_score = score[top_pc]

        data[contact]['top_pc'] = top_pc

        data[contact]['loading_score'] = top_score

        data[contact]['color'] = f'0x{chacra_colors[top_pc-1][1:-2]}'

        # take the slope by default from first 7 temps or length of df if it's smaller than 7 rows
        data[contact]['slope'] = get_slope(cdf,
                                        contact, 
                                        temp_range=(slope_range[0],
                                                    min(slope_range[1],cdf.shape[0])))

        # TODO I don't think this is being used anywhere....
        data[contact]['sel_1'] = f'resname {resna} and resnum {resia} and chain {chaina}'
        data[contact]['sel_2'] = f'resname {resnb} and resnum {resib} and chain {chainb}'

        # option to depict the temperature sensitivity rank in terms of how solid or transparent the spheres are
        # probably best when visualzing a single PC
    if variable_sphere_transparency:
        scores = []
        for contact in data.keys():
            scores.append(data[contact]['loading_score'])

        max_rank = max(scores)
        min_rank = min(scores)
        m = -(max_transparency/(max_rank-min_rank))
        b = -m

        for contact in data.keys():
            rank = data[contact]['loading_score']
            sphere_transparency = (m * rank) + b
            data[contact]['sphere_transparency'] = sphere_transparency

        # sort the dictionary in ascending order of loading score so that
        # the highest scores get colored last (and take visual precedence)
        # can also make the option to sort by any of the dictionary items
        # sort the dictionary by score
    result = collections.OrderedDict(sorted(data.items(), key=lambda t:t[1]["loading_score"]))
    return result

def write_group_selections(contact_data, output_file, ca_only=True):

    ###################
    # collect the data to produce group selections
    chacra_ids = set()
    for val in contact_data.values():
        chacra_ids.add(val['top_pc'])
    # hold the selection names under each pc id    
    chacra_selections = {pc:'' for pc in chacra_ids}
    # hold the line ids under each 
    lines = {pc:'' for pc in chacra_ids}
    #####################


    with open(output_file, 'w') as f:
        # iterating through dictionary and taking the contact name (contact)
        # and the corresponding dictionary (data) 

        for contact, data in contact_data.items():

            # name the selection
            contact = f"{data['chaina']}{data['resna']}{data['resia']}-{data['chainb']}{data['resnb']}{data['resib']}"
            if ca_only == True:
                # can't use group selections plus a modifier like "CA" so using this as workaround
                f.write(f"select {contact}, (chain {data['chaina']} and resi {data['resia']} "
                        f"or chain {data['chainb']} and resi {data['resib']}) and name CA\n")
            else:
                f.write(f"select {contact}, chain {data['chaina']} and resi {data['resia']} "
                        f"or chain {data['chainb']} and resi {data['resib']}\n")
            
            chacra_selections[data['top_pc']] += f'{contact} '

            f.write(f"color {data['color']}, {contact}\n")
            # draw the line
            f.write(f"distance {contact}-line, (chain {data['chaina']} and resi {data['resia']} and name CA), "
                   f"(chain {data['chainb']} and resi {data['resib']} and name CA)\n")

            lines[data['top_pc']] += f'{contact}-line '

            # set the line dash gap (just need to set it to 0 for positive slopes)
            # Can add the options for different line width and dash gaps here
            if data['slope'] >= 0.0:
                f.write(f"set dash_gap, 0, {contact}-line\n")

            if 'sphere_transparency' in data.keys():
                f.write(f"set sphere_transparency, {data['sphere_transparency']}, {contact} \n")

            # done with a contact's commands
            f.write('\n')
        f.write('#### Grouping commands ##### \n')
        for group in chacra_selections:
            f.write(f'group chacra_{group}, {chacra_selections[group]} \n')
            # add spheres
            f.write(f'show spheres, chacra_{group} \n')
            # color
            # TODO - coloring by groups means that the lowest pc colors overide 
            # everything else so it's no longer always highest loading score color
            #f.write(f'color 0x{chacra_colors[group-1][1:-2]}, chacra_{group} \n')
            
        for group in lines:
            f.write(f'group {group}_line, {lines[group]}\n')
            f.write(f'color 0x{chacra_colors[group-1][1:-2]}, {group}_line \n')








def write_selections(contact_data, output_file):
    '''
    Write the pymol commands for a depicting the contacts to a file.

    Parameters
    ----------
    contact_data : dictionary
        the get_contact_data() dictionary 

    output_file : str
        path to the output pml file
    

    '''


    with open(output_file, 'w') as f:
        # iterating through dictionary and taking the contact name (contact)
        # and the corresponding dictionary (data) 



        for contact, data in contact_data.items():

            # name the selection
            contact = f"{data['chaina']}{data['resna']}{data['resia']}-{data['chainb']}{data['resnb']}{data['resib']}"
            f.write(f"select {contact}, chain {data['chaina']} and resi {data['resia']} "
                    f"or chain {data['chainb']} and resi {data['resib']}\n")
            # color
            f.write(f"color {data['color']}, {contact}\n")

            # draw the line
            f.write(f"distance {contact}-line, (chain {data['chaina']} and resi {data['resia']} and name CA), "
                   f"(chain {data['chainb']} and resi {data['resib']} and name CA)\n")
            
            # color the line 
            f.write(f"color {data['color']}, {contact}-line\n")

            # set the line dash gap (just need to set it to 0 for positive slopes)
            # Can add the options for different line width and dash gaps here
            if data['slope'] >= 0.0:
                f.write(f"set dash_gap, 0, {contact}-line\n")

            # show spheres
            f.write(f"show spheres, {contact} and name CA\n")

            if 'sphere_transparency' in data.keys():
                f.write(f"set sphere_transparency, {data['sphere_transparency']}, {contact} \n")

            # done with a contact's commands
            f.write('\n')

        


def to_pymol(contact_list, contactFrequencies, contactPCA,
                     output_file = 'output.pml',
                    slope_range=(0,7),
                    pc_range=(1,4),
                    variable_sphere_transparency=False,
                    group=True):
    '''
     Parameters
    ----------
    contact_list : list
        List of the contacts in format chain1:resname1:resnum1-chain1:resname1:resnum1
    
    contactFrequencies : ContactFrequencies object

    contactPCA : contactPCA object

    slope_range : tuple of int
        The lowest and highest (row) indices (inclusive) from the contact data to calculate
        the slope of the contact frequencies within

    pc_range : tuple of in
        The lowest and highest PCs (inclusive and 1 indexed) to consider when
        identifying the highest loading scores, coloring, etc.

    Usage
    -----
    max_pc = 7
    top_contacts = []
    for i in range(1,max_pc+1):
        top_contacts.extend(cpca.sorted_norm_loadings(i).index[:20])
    top_contacts = list(set(top_contacts))
    to_pymol(top_contacts, ContactFrequencies, ContactPCA, output_file, pc_range(1,max_pc))
    '''

    # remove any duplicates
    contact_list = list(set(contact_list))
    # get all the data
    contact_data = get_contact_data(contact_list, contactFrequencies, contactPCA,
                    slope_range=slope_range,
                    pc_range=pc_range,
                    variable_sphere_transparency=variable_sphere_transparency
                    )
    # write it to pymol file
    if output_file.split('.')[-1] != 'pml':
        print("Your output file must have '.pml' appended for pymol to run it.")
    if group == True:
        write_group_selections(contact_data, output_file)
    else:
        write_selections(contact_data, output_file)




def get_slope(df,contact,temp_range=(0,7)):
    '''
    Return the slope of the contact frequency within the specified temperature
    (index) range.

    df: pd.DataFrame of contact frequencies 

    contact: str
        The name of the contact to retrieve the frequency slope for.
    
    temp_range: 2 int tuple
        The dataframe's index range to retrieve the frequency slope within.
    
    '''

    return linregress(df[contact].iloc[temp_range[0]:temp_range[1]].index, 
                   df[contact].iloc[temp_range[0]:temp_range[1]]).slope



def get_variance_to_sphere_scale_interpolator(eigenvalues,
                                 min_sphere_scale=0.6,
                                 max_sphere_scale=1.2,
                                 pcs=[i+1 for i in range(3)]):
    '''
    Take an np.array of the eigenvalues corresponding to the PCs you want to
    adjust the sphere sizes for and return a scipy interpolation function that
    will let you map the ranks on a given pc to a sphere size.

    Parameters
    ----------
    eigenvalues : np.array
        eigenvalue array
        
    max_sphere_scale: maximum size you want the largest eigenvalue spheres to 
    be in pymol
    
    min_sphere_scale: minimum sphere size to depict in pymol
    Returns
    -------
    scipy interpolater.

    '''
    variance = eigenvalues/eigenvalues.sum()
    # principal component id corresponding to the last one you're going
    # to depic contacts for
    max_pc = int(max(pcs)-1)
    min_variance = min(variance[:max_pc])
    max_variance = max(variance[:max_pc])
    
    interpolator = interp1d([min_variance, max_variance],
                            [min_sphere_scale, 
                             max_sphere_scale])
    
    return interpolator, variance

def get_sphere_scale(interpolator, pc, variance):
    '''
    Take the interpolator and explained variance from the above function
    and return the sphere scale for depicting the contact.

    Parameters
    ----------
    interpolator : scipy interp1d object
        DESCRIPTION.
    pc : int
        pc of contact.
    variance : np.array
        array of explained variance by pc.

    Returns
    -------
    float corresponding to the size of the sphere to depict in pymol.

    '''
    return interpolator(variance[pc-1])



def pymol_averaged_chacras_to_all_subunits(mapped_contacts, pymol_data, output):
    '''
    Create data for contacts_to_pymol.write_selections so that you can visualize the averaged contact
    data's top PC contacts (chacras) applied back to all the subunits from which they were averaged.
    
    mapped_contacts : dictionary
        dictionary containing the averaged contact name keys and list values that contain the equivalent
        contact name for the other subunits. Generate this with everything_from_averaged(as_map=True).
    
    pymol_data : dictionary
        dictionary of all the data needed for write_selections.  Generate with contacts_to_pymol.get_contact_data
        
    Returns
    -------
    Dictionary of contact data for input to write_selections.  
    
    Usage
    -----
    max_pc = 7
    top_contacts = []
    for i in range(1,max_pc+1):
        top_contacts.extend(cpca.sorted_norm_loadings(i).index[:20])
    top_contacts = list(set(top_contacts))
    mapped_contacts = everything_from_averaged(avg_contact_df[top_contacts], all_contact_frequency_df, 
                                    mda.universe, ['A','G'],as_map=True)
    # ['A','G'] is the list of representative chains used when generating averaged_contacts
    
    pymol_data = get_contact_data(mapped_contacts.keys(),average_ContactFrequencies,ContactPCA, pc_range=(1,max_pc))

    pymol_averaged_chacras_to_all_subunits(mapped_contacts, pymol_data, 'path/to/output.pml')
    
    '''
    full_protein_pymol_data = {}
    for averaged_contact_name in pymol_data:
        full_protein_pymol_data[averaged_contact_name] = pymol_data[averaged_contact_name]
        for replicated_name in mapped_contacts[averaged_contact_name]:
            if replicated_name != averaged_contact_name:
                resids = parse_id(replicated_name)
                full_protein_pymol_data[replicated_name] = {
                            'chaina': resids['chaina'],
                            'resna': resids['resna'],
                            'resia': resids['resida'],
                            'chainb': resids['chainb'],
                            'resnb': resids['resnb'],
                            'resib': resids['residb'],
                            'top_pc': pymol_data[averaged_contact_name]['top_pc'],
                            'loading_score': pymol_data[averaged_contact_name]['loading_score'],
                            'color': pymol_data[averaged_contact_name]['color'],
                            'slope': pymol_data[averaged_contact_name]['slope'],
                            'sel_1': f"resname {resids['resna']} and resnum {resids['resida']} and chain {resids['chaina']}",
                            'sel_2': f"resname {resids['resnb']} and resnum {resids['residb']} and chain {resids['chainb']}"
                            }
                
    write_group_selections(full_protein_pymol_data, output)


###### GRADIENT COLORING #################
        
def contact_frequency_color_gradient(output='./colors.pml',df=None, 
                                     contact_pca=None, pc=1,
                                     cmap_name='plasma', contact_list=None):
    '''
    Find top scoring contact for each residue and color according to slope
    Probably need to put a data prep function in ContactFrequencies and feed
    the data to this.
    '''
    
    residue_slopes = {}
    loading_scores = {}
    
    for contact in contact_list:
        resa, resb = contact.split('-')
        cha, resna, resnuma = resa.split(':')
        chb, resnb, resnumb = resb.split(':')
        slope = get_slope(df,contact)
        loading_score = contact_pca.get_top_score(contact, (pc,pc))
        loading_score = loading_score[loading_score.keys()[0]]
        
        if resa in list(residue_slopes.keys()):
            continue
        else:
            residue_slopes[resa] = slope
            loading_scores[resnuma] = loading_score
        if resb in list(residue_slopes.keys()):
            continue
        else:
            residue_slopes[resb] = slope
            loading_scores[resnumb] = loading_score

    
    #steepest_up = max(list(residue_slopes.values()))
    #steepest_down = min(list(residue_slopes.values()))
    
    n_residues = len(list(residue_slopes.keys()))
    cmap = cm.get_cmap(cmap_name, n_residues)
    #values = np.array(list(residue_slopes.values()))
    # How to deal with normalizing the negative number but keep the sign for 
    # diverging colormap?        
    #normalized_values =  (values-min(values))/(max(values)-min(values))
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    # need to use loading score instead of slope to assign colors
    with open(output, 'w') as file:
        for i, residue in enumerate(list(residue_slopes.keys())):
            chain, resn, resi = residue.split(':')
            if residue_slopes[residue] <= 0:
                index = norm(-loading_scores[resi])
                color_index = int(index *cmap.N)
            else:
                index = norm(loading_scores[resi])
                color_index = int(index *cmap.N)
            
                
            hex_color = '0x'+str(to_hex(cmap(color_index))[1:])
            
            file.write('color ' + hex_color + ', resi ' + str(resi) + \
                       ' and chain ' + chain + '\n' \
                       )
    
def slope_color_gradient(output,
                         contact_slope_dictionary, 
                         cmap_name="plasma"):
    '''
    Provide a dictionary of contact keys and corresponding slopes
    and color according to absolute value with solid or dashed line to 
    denote increasing or decreasing slope
    '''
    # sort the dictionary
    sorted_dict = dict(sorted(contact_slope_dictionary.items(), 
                              key=lambda item: np.abs(item[1]))) 
    # get an array of the slopes
    values = np.array(list(sorted_dict.values()))
    
    # count the residues that will be colored
    residues = []
    
    for contact,slope in sorted_dict.items():
        resa, resb = contact.split('-')
        cha, resna, resnuma = resa.split(':')
        chb, resnb, resnumb = resb.split(':')
    
        if resa not in residues:
            residues.append(resa)
        if resb not in residues:
            residues.append(resb)
    
    # make the cmap across the gradients
    # and make the range that will be used for the cmap indices with norm
    n_residues = len(residues)
    cmap = cm.get_cmap(cmap_name, n_residues) 
    norm = mpl.colors.Normalize(vmin=np.min(np.abs(values)), 
                                vmax=np.max(np.abs(values)))
    
    with open(output, 'w') as file:
        for i, contact in enumerate(list(sorted_dict.keys())):
            ch1,resn1,resi1,ch2,resn2,resi2 = re.split(':|-',contact)
            
            index = norm(np.abs(sorted_dict[contact]))
            color_index = int(index *cmap.N)
            
                
            hex_color = '0x'+str(to_hex(cmap(color_index))[1:])
            line_string = label_distance(ch1,resi1,ch2,resi2)
            if sorted_dict[contact]>0:
                dash_gap = 0
            else:
                dash_gap = 0.8
            file.write(f'color {hex_color}, (resi {resi1} and chain {ch1})\
   or (resi {resi2} and chain {ch2})\n\
   show spheres, ((resi {resi1} and chain {ch1})\
   or (resi {resi2} and chain {ch2})) and name CA\n\
   {line_string}\n\
   set dash_gap,{dash_gap},{resi1}-{resi2}\n\
   color {hex_color}, {resi1}-{resi2}\n'
                        )
    



################## NETWORKX ###############
def nx_edges_to_pymol(output, edges, color='blue', variable_width=False, max_weight=None, min_weight=None, max_width=20, min_width=2):
    #TODO add option to take first n from the full dictionary so you don't have to presort them
    '''
    make distance line depictions of networkx edges

    output : str
        path to output file
    
    edges : dictionary
        in the form of {tuple of nodes : edge weight, }
        {('A:MET:419', 'A:LEU:357'): 0.20897370785638944,
        ('A:TYR:359', 'A:LEU:357'): 0.18124780247685277,
        }
    
    color : str
        specify a pymol color string

    variable_width : bool
        adjust size of line based on edge weight

    max_weight : int or float
        specify the maximum edge weight used to normalize the weights in edges
        if None, max_weight is taken from max(edges.values())
    
    max_weight : int or float
        specify the minimum edge weight used to normalize the weights in edges
        if None, min_weight is taken from min(edges.values())
    
    max_width : int or float
        maximum dash width for highest edge weight

    min_width : int or float
        minimum dash width for lowest edge weight
    
    '''
    with open(output, 'w') as file:
        
        if variable_width:
            if max_weight == None:
                max_weight = max(edges.values())
            if min_weight == None:
                min_weight = min(edges.values())
            norm_denominator = max_weight - min_weight
            size_coef = max_width - min_width

        
        for edge in edges.keys():
            edge1, edge2 = edge[0], edge[1]
            edge1_chain, edge1_resn, edge1_resid = edge1.split(':')
            edge2_chain, edge2_resn, edge2_resid = edge2.split(':')
          
            name = f'{edge1_chain}{edge2_chain}-{edge1_resid}-{edge2_resid}-line'
            # draw the line
            dist_string = f"distance {name}, (chain {edge1_chain} and resi {edge1_resid} and name CA), "\
                   f"(chain {edge2_chain} and resi {edge2_resid} and name CA)\n"
            file.write(dist_string+'\n')

            color_string = f'color {color}, {name} \n'
            file.write(color_string)

            if variable_width:
                edge_weight = edges[edge]
                edge_size = (((edge_weight-min_weight)/(norm_denominator))*size_coef) + min_width
                width_string = f'set dash_width, {edge_size}, {name} \n'
                file.write(width_string)
     
    
    
def nx_to_pymol(output, sel_name, res_list, iteration):
    '''
    color network clusters using this in a loop that iterates over a list of
    clusters
    '''
    
    colors = ['red','blue','green','yellow','purple','white','orange','black']
    
    with open(output, 'a') as file:
    
        sel_string = 'select community_'+str(iteration)+ ','
        for i, res in enumerate(res_list):
            chain, resname, resid = res.split(':')
            if i != len(res_list) - 1:
                sel_string += '(resi '+resid+' and chain '+chain+') or '
            else:
                sel_string += '(resi '+resid+' and chain '+chain+')'
        
        if iteration > len(colors)-1:
            cycle = int(iteration/len(colors))
            iteration = iteration - (len(colors)*cycle)
        color_string = 'color '+colors[iteration]+', '+'community_'+str(iteration)
        
        pymol_input = sel_string +'\n'+ color_string +'\n'
        
        file.write(pymol_input)
        
def nx_path_to_pymol(output, sel_name, res_list):
    '''
    Generate a path between two residues and draw lines and spheres between
    the residues in the path
    '''
    with open(output, 'a') as file:
    
        sel_string = 'select '+sel_name+', '
        for i, res in enumerate(res_list):
            chain, resname, resid = res.split(':')
            if i != len(res_list) - 1:
                sel_string += '(resi '+resid+' and chain '+chain+') or '
                if i > 0:
                    # TODO the write_selections function replaced the label_distance function
                    # TODO add replacement functions
                    prev_chain, _, prev_resi = res_list[i-1].split(':')

    
                    dist_string = label_distance(chain, resid, prev_chain, 
                                                 prev_resi)
                    color_distance = 'color blue, '+resid+'-'+prev_resi+' \n'
                    file.write(dist_string+'\n')
                    file.write(color_distance)
            else:
                sel_string += '(resi '+resid+' and chain '+chain+')'
                prev_chain, _, prev_resi = res_list[i-1].split(':')
    
                dist_string = label_distance(chain, resid, prev_chain, 
                                             prev_resi)
                color_distance = 'color blue, '+resid+'-'+prev_resi+' \n'
                file.write(dist_string+'\n')
                file.write(color_distance)
                
        
        
        file.write(sel_string + ' \n')
        file.write('color blue, '+sel_name+' \n')
        file.write('show spheres, '+sel_name+' and name CA \n')
        
                

        
    

def nx_to_pymol_gradient(output, res_list, colors):
    '''
    take a res list (in the contact name chain:resname:resid format
    and provide a set of corresponding hex color codes to write a pymol
    file that will apply the colors to all the selections.
    '''
    
    with open(output, 'a') as file:
    
        for i, res in enumerate(res_list):
            chain, resname, resid = res.split(':')
            
            color_string = 'color '+str(colors[i])+', resi ' +str(resid)+ \
                        ' and chain '+chain
        
            pymol_input = color_string +'\n'
        
            file.write(pymol_input)
            
            
            
def node_centrality_gradient(output='./colors.pml',
                             cmap_name='plasma',node_centralities=None):
    '''
    Color residues according to their node centrality 
    node_centralities is dictionary output from nx.betweeness_centrality()
    '''
    
    
    with open(output, 'w') as file:
        values = np.array(list(node_centralities.values()))
            
        normalized_values =  (values-min(values))/(max(values)-min(values))
        
        n_residues = len(node_centralities)
            
        cmap = cm.get_cmap(cmap_name, n_residues)  
        
        
        for i, key in enumerate(list(node_centralities.keys())):
            chain, resn, resi = key.split(':')
            color_index = int(normalized_values[i] *cmap.N)
            hex_color = '0x'+str(to_hex(cmap(color_index))[1:])
            file.write('color ' + hex_color + ', resi ' + resi + \
                       ' and chain ' + chain + '\n' \
                       )
        


##################### GENERAL STRUCTURE COLORING ##################
#TODO get rid of pdb parser and switch to mdanalysis pdb functions
def coordinate_color_gradient(output='./colors.pml',structure=None,cmap_name='plasma', coord='y'):
    '''
    Produce a pymol coloring script to color a structure in a gradient
    across the given axis
    '''
    parser = PDBParser()
    structure = parser.get_structure('test', structure)
    with open(output, 'w') as file:
  
    # need to add checks for this line to make sure it's selecting the right model
        model = structure[0]
        for chain in model.get_chains():
            chain_id = chain.get_id()
            n_residues = len(model[chain_id].get_unpacked_list())
            coordinates = {}
            for residue in model[chain_id].get_residues():
                resnum = str(residue.id[1])
                for atom in residue.get_atoms():
                    if atom.name == 'CA':
                            if coord=='y':
                                coordinate = atom.coord[1]
                            elif coord=='x':
                                coordinate = atom.coord[0]
                            elif coord=='z':
                                coordinate = atom.coord[2]
                            coordinates[(resnum, chain_id)] = coordinate
        
            values = np.array(list(coordinates.values()))
            
            normalized_values =  (values-min(values))/(max(values)-min(values))
            
            cmap = cm.get_cmap(cmap_name, n_residues)  
            for i, resi in enumerate(list(coordinates.keys())):
                color_index = int(normalized_values[i] *cmap.N)
                hex_color = '0x'+str(to_hex(cmap(color_index))[1:])
                file.write('color ' + hex_color + ', resi ' + str(resi[0]) + \
                           ' and chain ' + resi[1] + '\n' \
                           )

        

#TODO Fix this function so it's more robust - the draft is at the bottom      
def circular_color_gradient(output='./colors.pml',structure=None,
                            cmap_name='plasma', coords=('x','y'), 
                            seg_id=None,extra_id=None):
    '''
    Normalize x and y from -1 to 1 and then find the length of the normalized 
    vector for each x,y position and color according to the % of max vector 
    length.
    '''
    parser = PDBParser()
    structure = parser.get_structure('test', structure)
    with open(output, 'w') as file:
  
    # need to add checks for this line to make sure it's selecting the right model
        model = structure[0]
        for chain in model.get_chains():
            chain_id = chain.get_id()
            if chain_id == seg_id:
                n_residues = len(model[chain_id].get_unpacked_list())
                coordinates = {}
                for residue in model[chain_id].get_residues():
                    resnum = str(residue.id[1])
                    # get the xyz coordinates of the first atom in the residue
                    coordinates[(resnum, chain_id)]=list(residue.get_atoms()
                                                    )[0].coord
                
            
                values = np.array(list(coordinates.values()))
                # center the data
                xcenter = values[:,0] - values[:,0].mean()
                
                ycenter = values[:,1] - values[:,1].mean()
                
                zcenter = values[:,2] - values[:,2].mean()
                
                centered_array = np.vstack((xcenter,ycenter,zcenter)).T
                  
            
                vector_magnitudes = {}
                for i, resi in enumerate(list(coordinates.keys())):
                    
                    vector_magnitudes[resi] = np.linalg.norm(
                                                [centered_array[i][0],
                                                 centered_array[i][1]])
            
                max_vector = max(vector_magnitudes.values())
                
                cmap = cm.get_cmap(cmap_name, n_residues)
                for i, resi in enumerate(list(vector_magnitudes.keys())):
                    norm_vector_length = vector_magnitudes[resi]/max_vector
                    
                    color_index = int(norm_vector_length *cmap.N)
                    hex_color = '0x'+str(to_hex(cmap(color_index))[1:])
                    file.write('color ' + hex_color + ', resi ' + str(resi[0]) + \
                               ' and chain ' + resi[1] + '\n' \
                               )
                    
            # this is better
            '''
            coordinates = {}
            for res in u.residues:
                coordinates[(res.resnum, res.segid)] = res.atoms[0].position

            values = np.array(list(coordinates.values()))
            # center the data
            xcenter = values[:,0] - values[:,0].mean()

            ycenter = values[:,1] - values[:,1].mean()

            zcenter = values[:,2] - values[:,2].mean()

            centered_array = np.vstack((xcenter,ycenter,zcenter)).T


            vector_magnitudes = {}
            for i, resi in enumerate(list(coordinates.keys())):
                # determine which plane the gradient is going in here (second index)
                # xyz dictionary maps string to index 0-2
                vector_magnitudes[resi] = np.linalg.norm(
                                            [centered_array[i][0],
                                            centered_array[i][1]])

            max_vector = max(vector_magnitudes.values())

            cmap = cm.get_cmap(cmap_name, n_residues)
            for i, resi in enumerate(list(vector_magnitudes.keys())):
                norm_vector_length = vector_magnitudes[resi]/max_vector

                color_index = int(norm_vector_length *cmap.N)
                hex_color = '0x'+str(to_hex(cmap(color_index))[1:])
                file.write('color ' + hex_color + ', resi ' + str(resi[0]) + \
                        ' and chain ' + resi[1] + '\n' \
                        )
                    
                    '''
