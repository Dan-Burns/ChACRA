#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 09:43:39 2022

@author: dburns
"""


import re
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from scipy.stats import linregress
from ContactAnalysis.contact_functions import _parse_id, check_distance
import matplotlib as mpl

from pylab import cm
from matplotlib.colors import to_hex
from scipy.interpolate import interp1d

def _get_slope(df,contact,temp_range=(0,7)):
    #TODO for networkx should combine slope and some min or max freq (b)
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

def prepare_contact(contact,pc,contact_df,temp_range=(0,7),):
   
   
    line_slope = linregress(contact_df[contact].iloc[temp_range[0]:temp_range[1]].index, 
                       contact_df[contact].iloc[temp_range[0]:temp_range[1]]).slope
    if line_slope >= 0:
        slope = 1
    else:
        slope = -1
    return((contact,slope,pc))


'''                       
def prepare_contact(contact,pc,contact_df,temp_range=(0,7),
                    pc_variance=None,interpolator=None):
   
    
    #interpolator and variance are from get_variance_to_sphere_scale_interpolator
    
    line_slope = linregress(contact_df[contact].iloc[temp_range[0]:temp_range[1]].index, 
                       contact_df[contact].iloc[temp_range[0]:temp_range[1]]).slope
    if line_slope >= 0:
        slope = 1
    else:
        slope = -1
    # Need a better way of implementing this - doesn't like the pc_variance=None
    # for calls that don't use sphere_variance
    # if you want to have different sphere sizes for different pcs
    if type(pc_variance) != None:
        sphere_scale = get_sphere_scale(interpolator, pc, pc_variance)
        return ((contact,slope,pc,sphere_scale ))
        
    else:
        return((contact,slope,pc))
'''

def write_selection(chain1, resname1, resid1, chain2, resname2, resid2):
    '''
    Write a pymol selection from chain and residue input
    '''
    
    selection = ('select ' + resname1 + resid1 + 'to' + resname2 + resid2 + ', chain ' + chain1 + ' and resi ' 
                 + resid1 + ' or chain ' + chain2 + ' and resi ' + resid2)
    
    return selection

def show_spheres(selection,scale=None):
     #selection = re.split(' |,', selection)[1]
     if scale:
         return 'show spheres, ' + selection + ' and name CA\n\
                 set sphere_scale,'+str(scale)+','+selection
     else:
         return 'show spheres, ' + selection + ' and name CA'
 
def color_selection_by_pc(selection, contact_tuple):
    '''
    Take pymol selection  and contact frequency line and write a pymol color command that colors according
    to contact frequency
    
    contact tuple can be generated from prepare_contact()
    
    Need to add an argument to include a cmap and specify gradients
        something like np.linspace(min-max_freqs), cmap=plasma)
        and have it replace the values below.
    '''
    
    selection = re.split(' |,', selection)[1]
    
    if int(contact_tuple[2]) == 1:
        color = 'red'
    elif int(contact_tuple[2]) == 2:
        color = '0x02a8f8'
    elif int(contact_tuple[2]) == 3:
        color = 'ytterbium'
    elif int(contact_tuple[2]) == 4:
        color = 'purpleblue'
    elif int(contact_tuple[2]) == 5:
        color = 'gray30'
    elif int(contact_tuple[2]) == 6:
        color = 'magenta'
    elif int(contact_tuple[2]) == 7:
        color = '0xfad300'
    elif int(contact_tuple[2]) == 8:
        color = 'greencyan'
    elif int(contact_tuple[2]) == 9:
        color = 'gray50'
   
    else:
        color = 'yellow'
    
    
    color_string = 'color ' + color + ', ' + selection
    
    return color_string

def label_distance(chain1, resid1, chain2, resid2):
    '''
    Take chain and residue and write a pymol distance command to draw line between the alpha carbons
    of the two selections
    '''
    
    distance_string = ('distance ' + str(resid1) + '-' + str(resid2) + ', (chain ' + str(chain1) + ' and resi ' 
                      + str(resid1) + ' and name CA), (chain ' + str(chain2) + ' and resi ' + str(resid2) 
                      + ' and name CA)') 
    
    return distance_string

def set_dash_gap(resid1, resid2, dash_gap):
    return 'set dash_gap,' + str(dash_gap) + ',' + resid1+'-'+resid2  + '\n'

def sort_contacts_by_highest_score(prepared_contact_list,contact_pca):
    rank = {}
    for contact in prepared_contact_list:
        rank[contact] = contact_pca.get_scores(contact[0])[contact[2]-1]['score']
    rank = {k: v for k, v in sorted(rank.items(), key=lambda item: item[1])}  
    sorted_ranks = list(rank.keys())

    return sorted_ranks

def make_selections(output, 
                    by_pc=True, contact_list=None,
                    calpha_spheres=True,
                    sphere_variance=False):
    '''
    Take the prepared contact_list and make pymol selections based on chain and residue ID
    if by_pc=True, you must supply a dictionary dict[resnum] = output from ContactPCA.get_top_contact()
    Color the Selections according to contact frequency -- needs work since they all are making multiple contacts 
    Make pymol distance object to draw line between the contacting residues
    write the commands to output
    
    sphere_variance
    '''
    with open(output, 'w') as file:
        
        if by_pc==True:
            for contact in contact_list:
                #select the contacting residues
                chain1 = contact[0].split('-')[0].split(':')[0]
                resname1 = contact[0].split('-')[0].split(':')[1]
                resid1 = contact[0].split('-')[0].split(':')[2]

                chain2 = contact[0].split('-')[1].split(':')[0]
                resname2 = contact[0].split('-')[1].split(':')[1]
                resid2 = contact[0].split('-')[1].split(':')[2]
                if sphere_variance:
                    sphere_scale = contact[3]
                      
                selection = write_selection(chain1, resname1, resid1, 
                                            chain2, resname2, resid2)
                
                color_string = color_selection_by_pc(selection, contact)
                distance_label = label_distance(chain1, resid1, chain2, resid2)
                #customize_line = customize_distance_line(freq_change, max_freq, resid1, resid2) 
                color_distance_line = color_selection_by_pc(distance_label, contact)
                
                    
                
                if contact[1] >= 0:
                    dash_gap = 0
                    dash_gap_string = set_dash_gap(resid1, resid2, dash_gap)
                    if sphere_variance:
                        sphere_string = show_spheres(selection.split()[1],sphere_scale)
                        
                        pymol_input = selection + '\n' + color_string + '\n'\
                        + distance_label + '\n' + color_distance_line +  \
                        '\n' + dash_gap_string + \
                        sphere_string + '\n'
                        file.write(pymol_input)
                    else:   
                        pymol_input = selection + '\n' + color_string + '\n'\
                        + distance_label + '\n' + color_distance_line + \
                        '\n' + dash_gap_string + \
                        'show spheres, ('+ selection.split()[1] +') and name CA \n\n'
                    
                        file.write(pymol_input)
                else:
                    if sphere_variance:
                        sphere_string = show_spheres(selection.split()[1],sphere_scale)
                        pymol_input = selection + '\n' + color_string + '\n'\
                        + distance_label + '\n' + color_distance_line + '\n' \
                        + sphere_string + '\n'
                        file.write(pymol_input)
                    else:
                        pymol_input = selection + '\n' + color_string + '\n'\
                        + distance_label + '\n' + color_distance_line + \
                        '\n'+ 'show spheres, ('+ selection.split()[1] +') and name CA \n\n'
                        file.write(pymol_input)
        else:
            for contact in contact_list:
                #select the contacting residues
                chain1 = contact.split('-')[0].split(':')[0]
                resname1 = contact.split('-')[0].split(':')[1]
                resid1 = contact.split('-')[0].split(':')[2]

                chain2 = contact.split('-')[1].split(':')[0]
                resname2 = contact.split('-')[1].split(':')[1]
                resid2 = contact.split('-')[1].split(':')[2]
   
              
                selection = write_selection(chain1, resname1, resid1, 
                                            chain2, resname2, resid2)
                
                
                distance_label = label_distance(chain1, resid1, chain2, resid2)
                #customize_line = customize_distance_line(freq_change, max_freq, resid1, resid2) 
                
                
                
              
                pymol_input = selection + '\n' \
                + distance_label + '\n' \
                + '\n'+ 'show spheres, ('+ selection.split()[1] +') and name CA \n'
                file.write(pymol_input)
                
        
    file.close()
    
def nx_edges_to_pymol(output, edge_list):
    '''
    make distance line depictions of networkx edges
    '''
    with open(output, 'w') as file:
    
        
        for edge in edge_list:
            edge1, edge2 = edge[0], edge[1]
            edge1_chain, edge1_resn, edge1_resid = edge1.split(':')
            edge2_chain, edge2_resn, edge2_resid = edge2.split(':')
            dist_string = label_distance(edge1_chain, edge1_resid, 
                                         edge2_chain, edge2_resid)
            file.write(dist_string+'\n')
            color_distance = f'color blue, {edge1_resid}-{edge2_resid} \n'
            file.write(color_distance)
     
    
    
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

        
        
def contact_frequency_color_gradient(output='./colors.pml',df=None, 
                                     loading_score_obj=None, pc=1,
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
        slope = _get_slope(df,contact)
        loading_score = loading_score_obj.get_scores(contact, (pc,pc))[0]['score']

        
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
        
        
        
        
        
        
        
        
    
