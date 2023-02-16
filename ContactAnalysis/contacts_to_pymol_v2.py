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

def _get_slope(df,contact,temp_range=(0,7)):
    #TODO for networkx should combine slope and some min or max freq (b)
    return linregress(df[contact].iloc[temp_range[0]:temp_range[1]].index, 
                   df[contact].iloc[temp_range[0]:temp_range[1]]).slope
                          
def prepare_contact(contact,pc,contact_df,temp_range=(0,7)):
    line_slope = linregress(contact_df[contact].iloc[temp_range[0]:temp_range[1]].index, 
                       contact_df[contact].iloc[temp_range[0]:temp_range[1]]).slope
    if line_slope >= 0:
        slope = 1
    else:
        slope = -1
        
    return((contact,slope,pc))

def write_selection(chain1, resname1, resid1, chain2, resname2, resid2):
    '''
    Write a pymol selection from chain and residue input
    '''
    
    selection = ('select ' + resname1 + resid1 + 'to' + resname2 + resid2 + ', chain ' + chain1 + ' and resi ' 
                 + resid1 + ' or chain ' + chain2 + ' and resi ' + resid2)
    
    return selection

def show_spheres(selection):
     selection = re.split(' |,', selection)[1]
     return 'show spheres, ' + selection + ' and name CA'
 
def color_selection_by_pc(selection, contact_tuple):
    '''
    Take pymol selection  and contact frequency line and write a pymol color command that colors according
    to contact frequency
    
    contact tuple can be generated from ContactPCA.get_top_contact()
    
    Need to add an argument to include a cmap and specify gradients
        something like np.linspace(min-max_freqs), cmap=plasma)
        and have it replace the values below.
    '''
    
    selection = re.split(' |,', selection)[1]
    
    if int(contact_tuple[-1]) == 1:
        color = 'red'
    elif int(contact_tuple[-1]) == 2:
        color = '0x02a8f8'
    elif int(contact_tuple[-1]) == 3:
        color = 'ytterbium'
    elif int(contact_tuple[-1]) == 4:
        color = 'purpleblue'
    elif int(contact_tuple[-1]) == 5:
        color = 'orange'
    elif int(contact_tuple[-1]) == 6:
        color = 'magenta'
    elif int(contact_tuple[-1]) == 6:
        color = 'yellow'
   
    else:
        color = 'black'
    
    
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



def make_selections(output, 
                    by_pc=True, contact_list=None,
                    calpha_spheres=True):
    '''
    Take the prepared contact_list and make pymol selections based on chain and residue ID
    if by_pc=True, you must supply a dictionary dict[resnum] = output from ContactPCA.get_top_contact()
    Color the Selections according to contact frequency -- needs work since they all are making multiple contacts 
    Make pymol distance object to draw line between the contacting residues
    write the commands to output
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
   
              
                selection = write_selection(chain1, resname1, resid1, 
                                            chain2, resname2, resid2)
                
                color_string = color_selection_by_pc(selection, contact)
                distance_label = label_distance(chain1, resid1, chain2, resid2)
                #customize_line = customize_distance_line(freq_change, max_freq, resid1, resid2) 
                color_distance_line = color_selection_by_pc(distance_label, contact)
                
                
                if contact[1] >= 0:
                    dash_gap = 0
                    dash_gap_string = set_dash_gap(resid1, resid2, dash_gap)
                    pymol_input = selection + '\n' + color_string + '\n'\
                    + distance_label + '\n' + color_distance_line + '\n' \
                    + '\n' + dash_gap_string + '\n'+ \
                    'show spheres, ('+ selection.split()[1] +') and name CA \n'
                
                    file.write(pymol_input)
                else:
                    
                    pymol_input = selection + '\n' + color_string + '\n'\
                    + distance_label + '\n' + color_distance_line + '\n' \
                    + '\n'+ 'show spheres, ('+ selection.split()[1] +') and name CA \n'
                    file.write(pymol_input)
                    
        
    file.close()
    
def nx_edges_to_pymol(output, edge_list):
    '''
    make distance line depictions of networkx edges
    '''
    with open(output, 'a') as file:
    
        
        for edge in edge_list:
            edge1, edge2 = edge[0], edge[1]
            edge1_chain, edge1_resn, edge1_resid = edge1.split(':')
            edge2_chain, edge2_resn, edge2_resid = edge2.split(':')
            dist_string = label_distance(edge1_chain, edge1_resid, 
                                         edge2_chain, edge2_resid)
                                                 
            color_distance = f'color blue, {edge1_resid}-{edge2_resid}) \n'
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
    
        sel_string = 'select '+sel_name+', '
        for i, res in enumerate(res_list):
            chain, resname, resid = res.split(':')
            if i != len(res_list) - 1:
                sel_string += '(resi '+resid+' and chain '+chain+') or '
            else:
                sel_string += '(resi '+resid+' and chain '+chain+')'
        
        if iteration > len(colors)-1:
            cycle = int(iteration/len(colors))
            iteration = iteration - (len(colors)*cycle)
        color_string = 'color '+colors[iteration]+', '+sel_name
        
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
        
        
        
        
        
        
        
        
    