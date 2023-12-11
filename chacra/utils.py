from itertools import combinations
import pandas as pd
import numpy as np
import re
from MDAnalysis.analysis.distances import distance_array


def sort_dictionary_values(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: -item[1]))

def geometric_progression(tmin, tmax, n):
    '''
    

    Parameters
    ----------
    tmin : int or float
        lowest end of temperature range.
    tmax : int or float
        highest end of temperature range.
    n : int
        number of temperatures.

    Returns
    -------
    temps : list
        temperatures.

    '''
    
    t = lambda tmin, tmax, n, i: tmin*np.exp(i*np.log(tmax/tmin)/(n-1))
    
    temps = []
    
    for i in range(n):
        temps.append(t(tmin,tmax,n,i))
        
    return temps



def parse_id(contact):
    '''
    take the contact name (column id) and return a dictionary of
    the residue A descriptors and residue B descriptors
    '''
    chaina, resna, resida, chainb, resnb, residb = re.split(":|-", contact)
    return {'chaina':chaina, 'resna':resna, 'resida':resida,
             'chainb':chainb, 'resnb':resnb, 'residb':residb}

def split_id(contact):
    '''
    take the contact name and split it into its two residue parts
    returns a dictionary where 'resa' will contain 'CH:RES:NUM'
    '''
    resa, resb = re.split("-", contact)
    return {'resa':resa, 'resb':resb}


def find_identical_subunits(universe):
    '''
    determines which subunits are identical

    Parameters
    ----------
    universe: 
        mda.Universe from the structure that the contact
        data was calculated on.
    
    Returns
    -------
    Dictionary containing integer id keys and lists of identical subunits ids.
    '''
    # dictionary of segids: list/sequence of residues
    residues = {seg.segid: seg.residues.resnames for seg in universe.segments}
    segids = list(residues.keys())
    # make a square matrix that will be filled with True values for identical subunits
    array = np.zeros((len(segids),len(segids)),dtype=np.bool_)
    # every subunit is identical with itself
    np.fill_diagonal(array,True)
    # work with it as a df
    identical_table = pd.DataFrame(array, columns=segids, index=segids)

    # Go through all pairwise combinations of subunits
    for combo in combinations(segids,2):
        # not identical if lengths are different
        if len(residues[combo[0]]) != len(residues[combo[1]]):
            identical_table[combo[1]][combo[0]], identical_table[combo[0]][combo[1]] = False, False
        else:
            # catch anything that might have same number of residues but different sequences
            bool = np.all(np.equal(residues[combo[0]],residues[combo[1]]))
            # Enter True or False in both the upper tri and lower tri  
            identical_table[combo[1]][combo[0]], identical_table[combo[0]][combo[1]] =  bool, bool

    # Only keep one representative row for each unique chain that gives the identical sets of subunits
    identical_table.drop_duplicates(inplace=True)

    identical_subunits = {}
    # Add the lists of identical subunits to the dictionary
    # You can deal with different sets of identical subunits in complex situations\
    for i, segid in enumerate(identical_table.index):
        subunits = identical_table.T[identical_table.loc[segid]==True].index
        # if there is only one, it's just identical with itself
        if len(subunits) >= 2:
            identical_subunits[i] = sorted(list(subunits))

    return identical_subunits


def get_angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle
        


def get_chain_group(chain, identical_subunits):
    '''
    Return the identical subunit group that the subunit is in
    '''
    for group in identical_subunits:
        if chain in identical_subunits[group]:
            return group
        
def get_all_chain_dists(u):
    '''
    Minimum distances between each chain and all others
    This takes several seconds so run it before passing output to other functions
    
    '''

    segids = {segid for segid in u.segments.segids}
    sorted_all_chain_dists = {}
    all_chain_dists = {chain:{} for chain in segids}
    for chain in segids:
        sel1 = u.select_atoms(f'chainID {chain}')
        for chain2 in segids:
            if chain2 != chain:
                sel2 = u.select_atoms(f'chainID {chain2}')
                min_dist = distance_array(sel1.atoms, sel2.atoms,).min()
                all_chain_dists[chain][chain2] = min_dist
        sorted_all_chain_dists[chain]={k:v for k, v in sorted(all_chain_dists[chain].items(), key=lambda x:x[1])}

    return sorted_all_chain_dists



def make_equivalent_contact_regex(resids):
     '''
     resids : the parse_id dictionary containing the contact data
     #TODO can remove any potential ambiguity by adding the list of correct chain group chains 
     '''
     regex1 = f"[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)-[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)"
     regex2 = f"[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)-[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)"
     return f"{regex1}|{regex2}"

def get_representative_pair_name(chaina, chainb, identical_subunits, representative_chains, equivalent_interactions):
     '''
     provide the chains from the contact and get the chain names to use for making a generalized/averaged contact name
     '''
     if chaina == chainb:
          for key, identical_subunit_list in identical_subunits.items():
               if chaina in identical_subunit_list:
                    for representative_chain in representative_chains:
                         if representative_chain in identical_subunit_list:
                              representative_pair = (representative_chain, representative_chain)
                                   
     # determine which equivalent_interaction set it came from 
     else:
          paira = (chaina,chainb)
          
          for representative_pair_name, equivalent_interaction_list in equivalent_interactions.items():
               for pair in equivalent_interaction_list:
                    if paira == pair: #or pairb == pair:
                         representative_pair = representative_pair_name
                         
                         break
     
     return representative_pair

def get_pair_distance(sel1, sel2, u):
    '''
    Return the distance between two selections
    '''
    a = u.select_atoms(sel1).positions[0]
    b = u.select_atoms(sel2).positions[0]
    return np.linalg.norm(a-b)

def find_best_asymmetric_point(u, chain, all_chain_dists):
    '''
    Find the residue that creates the greates difference in distances between other chains
    by finding a point in the chain that's near some other neighboring chain

    #TODO can also just sample N=20? random points on the chain taken from evenly spaced points in 3D
        and take the one with the best differences
    '''
    A = np.where(u.atoms.segids == f'{chain}')[0]
    # asymmetric center of mass that is some point on the periphery of seg_combo[0] near seg_combo[1]
    # if they're not within 5 angstroms, this won't work - need to just identify closest chain, pick one at random
    # and then establish the as_com with that
    minimum_difference = {}
    for neighbor, distance in all_chain_dists[chain].items():
        if distance < 5:
            distances_to_chains = []
            as_com = u.select_atoms(f'(around 5 chainID {neighbor}) and chainID {chain}').center_of_mass()
            # the distances between all the atoms in seg_combo[0] and the asymmetric center of mass
            dists = np.apply_along_axis(np.linalg.norm, 1,u.atoms[A].positions - as_com)
            #find the closest res to the as_com so that this residue can be used for the reference point for other identical chains
            resid = u.atoms[A][np.where(dists==dists.min())[0][0]].resid
            A_pos = u.select_atoms(f'chainID {chain} and resnum {resid} and name CA').positions[0]
            for seg in u.segments.segids:
                if seg != chain:
                    B_com = u.select_atoms(f'chainID {seg}').center_of_mass()
                    distances_to_chains.append(np.linalg.norm(A_pos-B_com))
            minimum_difference[neighbor] = min([np.abs(i-j) for i, k in enumerate(distances_to_chains)
                                                for j, l in enumerate(distances_to_chains) if k !=l])
    
    return max(minimum_difference, key=minimum_difference.get)

def asymmetric_measurements(seg_combo, identical_subunits, u, all_chain_dists):
    '''
    This will identify all the equivalent interaction pairs as seg_combo
    all_chain_dists is included so the best residue is used to create the best variance in the measured distances 
    
    seg_combo : tuple(string, string)
        The pair of chains that you want to identify all the equivalent interacting pairs for.
        ('A','B') will find other identical chains pairs that interact with the same geometry as A and B
    
    identical_subunits : dictionary
        The dictionary with integer values and lists of identical subunits.

    all_chain_dists : dictionary
        The dictionary with chain_id values and nest dictionaries giving the chain keys and the minimum distance to them as values
    
    Returns
    -------
    list of tuples
    each tuple will be a pair of chains with equivalent interaction geometry
    
    '''

    # whole protein center of mass
    com = u.select_atoms('protein').center_of_mass()
    # indices of atoms corresponding to chain seg_combo[0]
    A = np.where(u.atoms.segids == f'{seg_combo[0]}')[0]
    # asymmetric center of mass that is some point on the periphery of seg_combo[0] near seg_combo[1]
    # if they're not within 5 angstroms, this won't work - need to just identify closest chain, pick one at random
    # and then establish the as_com with that
    neighbor = find_best_asymmetric_point(u, seg_combo[0], all_chain_dists)
    as_com = u.select_atoms(f'(around 5 chainID {neighbor}) and chainID {seg_combo[0]}').center_of_mass()
    # the distances between all the atoms in seg_combo[0] and the asymmetric center of mass
    dists = np.apply_along_axis(np.linalg.norm, 1,u.atoms[A].positions - as_com)
    #find the closest res to the as_com so that this residue can be used for the reference point for other identical chains
    resid = u.atoms[A][np.where(dists==dists.min())[0][0]].resid
    # better to just use CA instead of this atom for distance measurements since there's more variability in sidechain positions
    atom = u.atoms[A][np.where(dists==dists.min())[0][0]].name
    A_pos = u.select_atoms(f'chainID {seg_combo[0]} and resnum {resid} and name CA').positions[0]
    # center of mass of seg_combo[1]
    B_com = u.select_atoms(f'chainID {seg_combo[1]}').center_of_mass()
    # this identifies A's relationship to B. 
    comparison_dist = np.linalg.norm(A_pos-B_com)
    comparison_angle = np.rad2deg(get_angle(A_pos,com,B_com))
     
    for key, seg_list in identical_subunits.items():
        if seg_combo[0] in seg_list and seg_combo[1] in seg_list:
            A_group, B_group = key, key
        elif seg_combo[0] in seg_list and seg_combo[1] not in seg_list:
            A_group = key
        elif seg_combo[1] in seg_list and seg_combo[0] not in seg_list:
            B_group = key

    relationships = []
    for seg1 in identical_subunits[A_group]:
        distances = {}
        distance_difs = {}
        angle_difs = {}
        for seg2 in identical_subunits[B_group]:
            if seg1 != seg2:
            
                pos1 = u.select_atoms(f'chainID {seg1} and resnum {resid} and name CA').positions[0]
                pos2 = u.select_atoms(f'chainID {seg2}').center_of_mass()
                distance = np.linalg.norm(pos1-pos2)
                distances[(seg1,seg2)] = distance
                angle_difs[(seg1,seg2)] = np.abs(np.rad2deg(get_angle(pos1,com,pos2)) - comparison_angle)
                distance_difs[(seg1,seg2)] = np.abs(distance-comparison_dist)

        min_dist, min_angle = min(distance_difs, key=distance_difs.get), min(angle_difs, key=angle_difs.get)  
        if min_dist != min_angle:
            # sorting everything to ensure alphabetical order will align with contact naming scheme
            if ((distance_difs[min_dist]) + (angle_difs[min_dist])) < (distance_difs[min_angle]) + (angle_difs[min_angle]):
                relationships.append(tuple(sorted(min_dist)))
            else:
                relationships.append(tuple(sorted(min_angle)))
        else:       
            relationships.append(tuple(sorted(min_dist)))
    return relationships

def get_equivalent_interactions(representative_chains, u):
    '''
    For each chain in representative_chains, find all other identical chains' interaction partners
    that are equivalent to the representative_chains interactions with all the other chains in the protein.
    For instance, chain A's interaction with a subunit D on the other side of the protein might be equivalent to
    chain B's interaction with subunit E for a symmetric multimer.

    This is useful when averaging the contact frequencies of a multimer and determining the correct naming
    for the averaged contact record and to ensure it's depicted correctly when visualized.

    Parameters
    ----------
    representative_chains : list of strings
        A list with the names of the chain ids that you want to establish equivalent interactions for

    '''

    segids = u.segments.segids
    all_chain_dists = get_all_chain_dists(u)
    identical_subunits = find_identical_subunits(u)

    equivalent_interactions = {}
    for chain in [chain for chain in representative_chains]:
        for segid in segids:
            if segid != chain:
                equivalent_interactions[tuple(sorted((chain,segid)))] = asymmetric_measurements((chain,segid),identical_subunits,u, all_chain_dists)

    return equivalent_interactions