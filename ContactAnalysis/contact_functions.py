#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 02:33:19 2022

@author: dburns
"""
import re
from Bio.PDB.PDBParser import PDBParser
import MDAnalysis as mda
import numpy as np




def _parse_id(contact):
    '''
    take the contact name (column id) and return a dictionary of
    the residue A identfiers and residue B identifiers
    '''
    chaina, resna, resida, chainb, resnb, residb = re.split(":|-", contact)
    return {'chaina':chaina, 'resna':resna, 'resida':resida,
             'chainb':chainb, 'resnb':resnb, 'residb':residb}

def check_distance(contact, structure):  ### Not being used - replaced with check_distance_mda
    '''
    This will compare distances between the same resids swapped between chain A and B to 
    find the combination representing the actual contact to depict on the pymol structure
    Would probably be faster with mdanalysis - just need to make mdanalysis selection syntax
    res.atoms.name('CA').position np.norm
    '''
    # in the class format can add structure=None to the class invocation and if its included can have the structure
    # object made outside of this function

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('test', structure)
    
    resids = _parse_id(contact)
    
    
    # need to add checks for this line to make sure it's selecting the right model
    model = structure[0]
    # can have the chain ids taken from the resids dictionary but the format is going to just be A and B for now
    # the next lines make the combinations of residue x on chain A and residue y on chain B 
    # and residue y on chain A and residue x on chain B
    chaina = model["A"]
    residueaa = chaina[int(resids['resida'])]
    residueab = chaina[int(resids['residb'])]
    atomaa = residueaa["CA"]
    atomab = residueab["CA"]
    chainb = model["B"]
    residuebb = chainb[int(resids['residb'])]
    residueba = chainb[int(resids['resida'])]
    atombb = residuebb["CA"]
    atomba = residueba["CA"]
    # This is assuming the original contact was always in the format of chain A as the first and chain B as second
    if atomaa-atombb < atomab-atomba:
        return contact
    else:
        return 'A:'+resids['resnb']+':'+str(resids['residb'])+'-'+\
            'B:'+resids['resna']+':'+str(resids['resida'])
                

def check_distance_mda(contact, u):
    '''
    This will compare distances between the same resids swapped between chain A and B to 
    find the combination representing the actual contact to depict on the pymol structure
    
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
    if np.linalg.norm((atomaa,atombb)) < np.linalg.norm((atomab,atomba)):
        return contact
    else:
        return 'A:'+resids['resnb']+':'+str(resids['residb'])+'-'+\
               'B:'+resids['resna']+':'+str(resids['resida'])
