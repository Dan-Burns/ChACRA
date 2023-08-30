#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 02:33:19 2022

@author: dburns
"""
import re
import MDAnalysis as mda
import numpy as np




def _parse_id(contact):
    '''
    take the contact name (column id) and return a dictionary of
    the residue A descriptors and residue B descriptors
    '''
    chaina, resna, resida, chainb, resnb, residb = re.split(":|-", contact)
    return {'chaina':chaina, 'resna':resna, 'resida':resida,
             'chainb':chainb, 'resnb':resnb, 'residb':residb}

def _split_id(contact):
        '''
        take the contact name and split it into its two residue parts
        returns a dictionary where 'resa' will contain 'CH:RES:NUM'
        '''
        resa, resb = re.split("-", contact)
        return {'resa':resa, 'resb':resb}

                
def check_distance_mda(contact, u, chain1='A', chain2='B'):
    '''
    This will compare distances between the same resids swapped between chain A and B to 
    find the combination representing the actual contact to depict on the pymol structure.

    Parameters
    ----------
    contact: string
        name of contact

    u: mda.Universe
        protein structure 

    chain1 and chain2 are the subunit names for the averaged contact, the chains will be swapped if
    residues in B-A contact are closer than the A-B contact.

    Returns
    -------
    contact name string.
    '''
    # in the class format can add structure=None to the class invocation and if its included can have the structure
    # object made outside of this function
    
   
    resids = _parse_id(contact)
    
    atomaa = u.select_atoms(f'chainid {resids["chaina"]} and resnum \
                            {resids["resida"]} and name CA').positions
    atombb = u.select_atoms(f'chainid {resids["chainb"]} and resnum \
                            {resids["residb"]} and name CA').positions
    # Switched chain and resids
    atomab = u.select_atoms(f'chainid {resids["chainb"]} and resnum \
                            {resids["resida"]} and name CA').positions
    atomba = u.select_atoms(f'chainid {resids["chaina"]} and resnum \
                            {resids["residb"]} and name CA').positions
    
    # use norm
    # This is assuming the original contact was always in the format of chain A as the first and chain B as second
    if np.linalg.norm((atomaa-atombb)) < np.linalg.norm((atomab-atomba)):
        return contact
    else:
        return f"{chain1}:{resids['resnb']}:{resids['residb']}-{chain2}:{resids['resna']}:{resids['resida']}"
