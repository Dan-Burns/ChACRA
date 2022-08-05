#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 01:51:26 2022

@author: dburns

Functions to make pymol selection files that you can drop into pymol with
your structure and visualize the contacts.
"""
import re

from scipy.stats import linregress

def write_selection(chain1, resname1, resid1, chain2, resname2, resid2):
    '''
    Write a pymol selection from chain and residue input
    '''
    
    selection = ('select ' + resname1 + resid1 + 'to' + resname2 + resid2 + ', chain ' + chain1 + ' and resi ' 
                 + resid1 + ' or chain ' + chain2 + ' and resi ' + resid2)
    
    return selection

def write_1_res_selection(chain, resname, resid):
    '''
    Write a pymol selection from chain and residue input
    '''
    
    selection = ('select ' + resname + resid + ', chain ' + chain + ' and resi ' 
                 + resid)
    
    return selection

def show_spheres(selection):
     selection = re.split(' |,', selection)[1]
     return 'show spheres, ' + selection + ' and name CA'

def color_selection_by_frequency(selection, contact):
    '''
    Take pymol selection syntax and contact frequency line and write a pymol color command that colors according
    to contact frequency
    frequency < 10 = blue
    10 < frequency < 25 = gold
    25 < frequency < 50 = purpleblue
    50 < frequency < 75 = ytterbium
    75 < frequency < 90 = red
    90 < frequency = white
    Need to add an argument to include a cmap and specify gradients
        something like np.linspace(min-max_freqs), cmap=plasma)
        and have it replace the values below.
    '''
    
    selection = re.split(' |,', selection)[1]
    
    if float(contact.split()[2]) < 0.1:
        color = '0x02a8f8'
    elif float(contact.split()[2]) < 0.25:
        color = 'gold'
    elif float(contact.split()[2]) < 0.5:
        color = 'purpleblue'
    elif float(contact.split()[2]) < 0.75:
        color = 'ytterbium'
    elif float(contact.split()[2]) < 0.9:
        color = 'red'
    else:
        color = 'white'
    
    
    color_string = 'color ' + color + ', ' + selection
    
    return color_string

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

def customize_distance_line(freq_change, max_freq, resid1, resid2):
    '''
    Take the convert_for_make_contacts "descriptive_dashes=True" 3 and 4th outputs and write the pymol
    syntax to make the dashes a particular thickness/radius and transparency
    '''
    # dash radius might need to be an exponential function
    dash_radius = 'set dash_radius, ' + str(0.5+float(freq_change)) + ', ' + resid1+'-'+resid2  + '\n'
    transparency = 'set dash_transparency, ' + str(1-float(max_freq)) + ', ' + resid1+'-'+resid2
    
    return dash_radius + transparency

def set_dash_gap(resid1, resid2, dash_gap):
    return 'set dash_gap,' + str(dash_gap) + ',' + resid1+'-'+resid2  + '\n'
    
def make_selections(output, contact_list=None, descriptive_dashes=False, 
                    by_pc=False, contact_dict=None,contact_df=None, n_temps=7,
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
            for res, contact in contact_dict.items():
                #select the contacting residues
                chain1 = contact[0].split('-')[0].split(':')[0]
                resname1 = contact[0].split('-')[0].split(':')[1]
                resid1 = contact[0].split('-')[0].split(':')[2]

                chain2 = contact[0].split('-')[1].split(':')[0]
                resname2 = contact[0].split('-')[1].split(':')[1]
                resid2 = contact[0].split('-')[1].split(':')[2]
                
                if str(res) == resid1:
                    selection = (chain1, resname1, resid1)
                else:
                    selection = (chain2, resname2, resid2)
                
                
                
                # color individual residues instead of both residues
                # make a function to check if a residue has already been colored
                # according to the highest pc it's on so it's not getting 
                # overwritten.
                selection = write_1_res_selection(selection[0], selection[1], 
                                                  selection[2])
                color_string = color_selection_by_pc(selection, contact)
                distance_label = label_distance(chain1, resid1, chain2, resid2)
                #customize_line = customize_distance_line(freq_change, max_freq, resid1, resid2) 
                color_distance_line = color_selection_by_pc(distance_label, contact)
                freq_change = linregress(contact_df[[contact[0]]][:n_temps].index, contact_df[contact[0]][:n_temps]).slope
                
                if freq_change >= 0:
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
        elif descriptive_dashes==True:
            # need to use the format from convert_for_make_contacts(descriptive_dashes=True)
            for contact in contact_list:
                #select the contacting residues
                chain1 = contact.split()[0].split(':')[0]
                resname1 = contact.split()[0].split(':')[1]
                resid1 = contact.split()[0].split(':')[2]

                chain2 = contact.split()[1].split(':')[0]
                resname2 = contact.split()[1].split(':')[1]
                resid2 = contact.split()[1].split(':')[2]
                freq_change = contact.split()[3]
                max_freq = contact.split()[4]

                selection = write_selection(chain1, resname1, resid1, chain2, resname2, resid2)
                color_string = color_selection_by_frequency(selection, contact)
                distance_label = label_distance(chain1, resid1, chain2, resid2)
                customize_line = customize_distance_line(freq_change, max_freq, resid1, resid2) 
                color_distance_line = color_selection_by_frequency(distance_label, contact)
                sphere_selection = show_spheres(selection)
                pymol_input = selection + '\n' + color_string + '\n' \
                + sphere_selection+ '\n' \
                + distance_label + '\n' + color_distance_line + '\n' +\
                customize_line + '\n'
                file.write(pymol_input)
            
        else:
            for contact in contact_list:
                #select the contacting residues
                chain1 = contact.split()[0].split(':')[0]
                resname1 = contact.split()[0].split(':')[1]
                resid1 = contact.split()[0].split(':')[2]

                chain2 = contact.split()[1].split(':')[0]
                resname2 = contact.split()[1].split(':')[1]
                resid2 = contact.split()[1].split(':')[2]
                
                # this is to fix the df column name to have the "-" in it again
                original_contact_name = contact.split()[0]+'-'+contact.split()[1]
                
                freq_change = linregress(contact_df[[original_contact_name]][:n_temps].index, contact_df[original_contact_name][:n_temps]).slope

                selection = write_selection(chain1, resname1, resid1, chain2, resname2, resid2)
                color_string = color_selection_by_frequency(selection, contact)
                distance_label = label_distance(chain1, resid1, chain2, resid2)
                color_distance_line = color_selection_by_frequency(distance_label, contact)
                if freq_change >= 0:
                    dash_gap = 0
                    dash_gap_string = set_dash_gap(resid1, resid2, dash_gap)
                    pymol_input = selection + '\n' + color_string + '\n'\
                    + distance_label + '\n' + color_distance_line + '\n' +\
                    dash_gap_string + '\n' + show_spheres(selection) + '\n'
    
                    file.write(pymol_input)
                else:
                    pymol_input = selection + '\n' + color_string + '\n'\
                    + distance_label + '\n' + color_distance_line + '\n' +\
                    show_spheres(selection) + '\n'
    
                    file.write(pymol_input)
                    
            
    file.close()
        
def convert_for_make_contacts(contact, freq=0.85, freq_gradient=False, contact_df=None, n_temps=7,
                             descriptive_dashes=False):
    '''
    remove hyphen from loading scores format and replace with space.
    also add a contact value of 0.85 to work with make_contacts
    Can use the freq_gradient option to color things according to whether they are increasing or 
    decreasing over first n_temps of contact_df
    '''
    contact1, contact2 = contact.split("-")
    
    if freq_gradient==True:
        slope = linregress(contact_df[[contact]][:n_temps].index, contact_df[contact][:n_temps]).slope
        if slope < 0:
            freq = 0.85
            dash = 'dash_gap,0.5,'
        else:
            freq = 0
            dash = 'dash_gap,0,'
    if descriptive_dashes==True:
        total_change = float(contact_df[[contact]][:n_temps].max()-contact_df[[contact]][:n_temps].min())
        max_frequency = float(contact_df[[contact]][:n_temps].max())
        return contact1 + " " + contact2 + " " + str(freq) + " " + str(total_change) + " " + str(max_frequency) + " " + str(dash)
    else:        
        return contact1 + " " + contact2 + " " + str(freq)
    
#to_visualize = list(map(convert_for_make_contacts, contacts_list))