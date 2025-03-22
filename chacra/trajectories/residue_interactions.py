from openmm.unit import *
import MDAnalysis as mda
from scipy.spatial.distance import pdist, squareform
from eidos.omm import *
import numpy as np
from MDAnalysis.analysis.distances import self_distance_array
from itertools import combinations
import argparse

'''
Not Implemented

Experimental. Alternative to binary contact definition. This will give you continuous
values for all pairwise interactions based on coulomb and LJ energy. Assuming only the protein
is used in the calculation, it's meant to provide a continuous set of values for ChACRA that 
reflects the chemistry specific to the interactions and not as an accurate representation 
of the system energy. 

Current form is very slow, particularly for large systems.

TODO: only consider interactions up to a cutoff
'''

enaught = (8.8541878128e-12)*((coulomb**2)/(joule*meter))
coulomb_constant = 1.602176634e-19

def get_nonbonded_force(openmm_system):
    nonbonded_force = None
    for force_object in openmm_system.getForces():
        if force_object.getName() == 'NonbondedForce':
            nonbonded_force = force_object
        
    return nonbonded_force

def get_particle_parameters(nonbonded_force, i):
    '''
    Separate the OpenMM nonbonded parameter tuple.

    i : int
        atom index

    Returns
    -------
    openmm.unit.Quantity
    charge, sigma, epsilon
    '''
    return nonbonded_force.getParticleParameters(i)


def get_all_params(openmm_system, indices):
    '''
    openmm_system : openmm.System

    indices : array-like
        The indices of the atoms from the system topology that you want the nonbonded parameters for.
    
    Returns
    -------
    The n_particle x 3 array of floats for charge (coulombs), sigma (nanometers), 
    and epsilon (kj/mol) parameters.
    '''

    nonbonded_force = get_nonbonded_force(openmm_system)
    n_atoms = len(indices)
    all_par = np.zeros((n_atoms, 3))
    for i in indices:
        q, sig, eps = get_particle_parameters(nonbonded_force, i)
        all_par[i] = [q.in_units_of(coulomb)._value, sig._value, eps._value]
    
    return all_par

def pair_ind_to_dist_ind(d, i, j):
    '''
    d : int
        number of atoms
    i : int
        row index
    j : int
        column index
    
    https://gist.github.com/CMCDragonkai/d663840fc151fca01e2bee242e792a3d
    '''
    index = d*(d-1)/2 - (d-i)*(d-i-1)/2 + j - i - 1
    return int(index)

def dist_ind_to_pair_ind(d, i):
    '''
    https://gist.github.com/CMCDragonkai/d663840fc151fca01e2bee242e792a3d
    '''
    b = 1 - 2 * d
    x = np.floor((-b - np.sqrt(b**2 - 8*i))/2).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    return (x,y)

def get_intra_res_indices(residues, atoms):
    '''
    Get the condensed distance vector of atom indices that represent intra-residue interactions
    '''
    n_atoms = len(atoms)
    intra_res_pairs = []
    for res in set(residues):
        intra_res_array_indices = np.where(residues == res)
        intra_res_atoms_indices = atoms[intra_res_array_indices]
        for combo in combinations(intra_res_atoms_indices, 2):
            intra_res_pairs.append(combo)
    condensed_intra_res_indices = []
    for pair in intra_res_pairs:
        condensed_intra_res_indices.append(pair_ind_to_dist_ind(n_atoms, pair[0], pair[1]))
    return condensed_intra_res_indices

def get_1_4_exclusions(nonbonded_force, indices):
    '''
    nonbonded_force : 

    indices : array-like
        The list of atom indices 

    Returns
    -------
    condensed distance vector for the the excluded atom index pairs (1-2 and 1-3 interactions) 
    condensed distance vector for the atom pairs that will have scaled interactions (1-4 interactions)
    array of scaled charges (these are squared) of same size as the scaled interactions condensed distance vector
    array of scaled sigmas (these are already treated by combining rules)
    array of scaled epsilons(already treated by combining rules)
    '''

    excluded_pairs = []
    scaled_pairs = []
    scaled_qs = []
    scaled_sigmas = []
    scaled_epsilons = []

    n_atoms = len(indices)
    for i in range(nonbonded_force.getNumExceptions()):
        ix1, ix2, q, sig, eps = nonbonded_force.getExceptionParameters(i)
        if (ix1 in indices) and (ix2 in indices): # only continuing if they're in the atom selection
            q = q.in_units_of(coulombs**2)._value
            if eps._value == 0: # not scaled, just excluded
                excluded_pairs.append((ix1,ix2))

            else: # not excluded, but scaled
                scaled_pairs.append([ix1,ix2])
                scaled_qs.append(q)
                scaled_sigmas.append(sig._value)
                scaled_epsilons.append(eps._value)

    # converted excluded_pairs indices tuples to list of single index values for condensed distance vector
    excluded_condensed_indices = []
    for tup in excluded_pairs:
        excluded_condensed_indices.append(pair_ind_to_dist_ind(n_atoms, tup[0], tup[1]))

    scaled_condensed_indices = []
    for tup in scaled_pairs:
        scaled_condensed_indices.append(pair_ind_to_dist_ind(n_atoms, tup[0], tup[1]))

    return excluded_condensed_indices, np.array(scaled_condensed_indices), np.array(scaled_qs), np.array(scaled_sigmas), np.array(scaled_epsilons)

def sigma_ij(sigma_i,sigma_j):
    return (sigma_i+sigma_j)/2

def eps_ij(eps_i, eps_j):
    return np.sqrt(eps_i*eps_j)

def make_precomputed_param_vectors(params, scaled_indices=None, scaled_params=None, param_type=None):
    '''
    params : np.array
        The 1d vector of parameters floats from get_all_params() corresponding to 
        param_type

    scaled_indices : np.array
        condesned distance vector of param pair indices that are scaled (1-4 interactions)
    
    scaled_params : np.array
        The n_atom length vecotr of scaled params - same type as param_type.

    param_type : str
        "charge", "sigma", or "epsilon"
        appropriate calculation will be applied.

    Returns
    -------
    np.array
    condensed distance vector of precomputed input parameters for potential energy functions
    '''

    n_atoms = params.shape[0]

    i_matrix = np.broadcast_to(params,(n_atoms,n_atoms))
    j_matrix = np.broadcast_to(params,(n_atoms,n_atoms))
    if param_type == 'charge':
        ij_matrix = i_matrix * j_matrix
    elif param_type == 'sigma':
        ij_matrix = sigma_ij(i_matrix, j_matrix)
    elif param_type == 'epsilon':
        ij_matrix = eps_ij(i_matrix, j_matrix)
    # make a square symmetric matrix of the precomputed params
    W = np.tril(ij_matrix) + np.triu(ij_matrix.T, 1)
    np.fill_diagonal(W, [0])
    W = squareform(W) # convert to condensed form
    W[scaled_indices] = scaled_params # set the scaled params 

    return W

def condensed_lj_potential(sigs, eps, r):
    '''
    Expects combined sigmas and epsilons (combination rules have already been applied)
    r is converted from angstroms to nanometers inside the function.

    Returns
    -------
    np. array
    Condensed distance vector of Lennard Jones potentials in kj/mol
    '''
    return 4*eps*(((sigs/(r*1e-1))**12)-((sigs/(r*1e-1))**6))

def condensed_coulomb_potential(q, r):
    '''
    Expects combined charges (q)
    r is converted from angstroms to meters inside the function
    
    Returns
    -------
    np.array
    Condensed distance vector of Lennard Jones potentials in kj/mol
    '''                               

    return ((AVOGADRO_CONSTANT_NA._value/1000)*q)/(4*np.pi*(r*1e-10)*enaught._value)

def condensed_lj_force(sig, eps, r):
    '''
    converts r from angstroms to nanometers as sig is expected to be in nanometers
    '''
    return -(48*eps)*((np.power(sig,12)/np.power(r*1e-1,13))-(0.5*(np.power(sig,6)/np.power(r*1e-1,7))))

def condensed_coulomb_force(q, r):
    '''
    converts r from angstroms to nanometers as sig is expected to be in nanometers
    '''
    return q/(4*np.pi**((r*1e-10)**2))

def get_trajectory_interaction_energies(u, qs, sigs, eps, selection, backend='serial'):
    '''
    backend : str
        Set to "OpenMP" to parallelize
    '''
    # this quickly gets way too big to hold in memory if you hold each frame

    sel = u.select_atoms(selection)
    all_dists = self_distance_array(sel)
    n_frames = len(u.trajectory)
    #energies = np.zeros((len(u.trajectory), all_dists.shape[0]))
    energies = np.zeros(all_dists.shape[0])
    for i, frame in enumerate(u.trajectory):
        sel = u.select_atoms(selection)
        all_dists = self_distance_array(sel, backend=backend)
        coulomb_energies = condensed_coulomb_potential(qs, all_dists)
        lj_energies = condensed_lj_potential(sigs, eps, all_dists)
        #energies[i] = coulomb_energies + lj_energies
        energies += (coulomb_energies + lj_energies)/n_frames
    return energies

def get_res_rows(res, resis):
    '''
    return the indices corresponding to a single residue
    to index that residue's atoms' rows from a n_atoms x n_atoms matrix
    '''
    return (np.where(resis==res)[0])[:,None]
    
def get_other_res_cols(res, resis):
    '''
    return the indices of the columns for all other residues' atoms
    '''
    return np.where(resis!=res)[0]

def make_residue_atoms_index_dictionary(resis):

    res_dict = {}
    for res in set(resis):
        res_dict[res] = (np.where(resis==res)[0])
    return res_dict

def get_pairwise_residue_energies(energies, mask, resis, atoms):
    '''
    Returns symmetric matrix of pairwise residue interaction energies
    '''
    # pairwise residue energies
    # mean_energies = energies.mean(axis=0) # will already be mean_energies
    mean_energies = energies
    mean_energies[mask] = 0
    mean_energies_array = squareform(mean_energies)
    
    res_atoms = make_residue_atoms_index_dictionary(resis,atoms)
    res_ixs = list(set(resis))
    res_to_rows = {res: get_res_rows(res, resis) for res in res_ixs}
    
    res_energies = np.zeros((len(set(resis)), len(set(resis))))

    for i, res in enumerate(res_ixs[:-1]):
        rows = res_to_rows[res]
        other_resis = res_ixs[i + 1:]
        for j, other_res in enumerate(other_resis, start=i+1):
            cols = res_atoms[other_res]
            res_energies[i,j] = mean_energies_array[rows,cols].sum(axis=1).sum(axis=0)
    #convert to a symmetric matrix
    ij_matrix = res_energies
    triu = np.triu(ij_matrix)
    W = triu + np.tril(triu.T, 1)
    return W

def make_column_names(pair_ids, pair_names, pair_chains):
    '''
    Make the contact dataframe column names
    ch:resname:resid-ch:resname:resid
    '''
    cols = []
    for i in range(len(pair_ids)):
        cha, chb = pair_chains[i]
        resna, resnb = pair_names[i]
        resia, resib = pair_ids[i]
        cols.append(f"{cha}:{resna}:{resia}-{chb}:{resnb}:{resib}")
    return cols

def get_condensed_contact_info(condensed_array, resids,
                                            resnames=None,
                                            chainids=None):
    '''
    condensed_array : np.array
        Condensed distance vector of pairwise residue energies

    resids : np.array
        The array of resids (these can be the canonical resids)
        These will be used in generating the column ids for the interaction dataframe
        If there are multiple chains with overlapping residue ids, the resids array
        shouldn't be generated from a set()...

        resids should only have 
    '''

    n_condensed_elements = condensed_array.shape[0]
    n_resis = len(resids)
    # tuples of pairwise indices
    indices = [dist_ind_to_pair_ind(n_resis, i) for i in range(n_condensed_elements)]
    
    condensed_resi_indices = np.array(indices)
    if (resnames is None) and (chainids is None):
        pair_ids = resids[condensed_resi_indices]
        return pair_ids
    else:
        pair_ids = resids[condensed_resi_indices]
        pair_names = resnames[condensed_resi_indices]
        pair_chains = chainids[condensed_resi_indices]
        return pair_ids, pair_names, pair_chains



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate interaction energies from an MD trajectory.")
    parser.add_argument('--topology', type=str, help="The pdb or topology file.")
    parser.add_argument('--trajectory', type=str, default=None, required=False, help="The MD trajectory.")
    parser.add_argument('--system', type=str, help="The path to the OpenMM system.")
    parser.add_argument('--selection', type=str, help="The MDAnalysis atoms selection string.")

    # Parse the arguments
    args = parser.parse_args()
    system = open_system(args.system)
    nonbonded_force = get_nonbonded_force(system)
    u = mda.Universe(args.topology, args.trajectory)
    sel_string = "protein"
    sel = u.select_atoms(sel_string)

    atoms = sel.atoms.ix
    resis = sel.atoms.resindices
    atom_names = sel.atoms.names
    resnames = sel.atoms.resnames

    all_par = get_all_params(system, atoms)
    all_dists = self_distance_array(sel.atoms)
    exclusion_indices, scaled_indices, scaled_qs, scaled_sigs, scaled_epsilons = get_1_4_exclusions(
        nonbonded_force, 
        atoms
    )

    qs = make_precomputed_param_vectors(all_par[:,0], 
                                        scaled_indices=scaled_indices,
                                        scaled_params=scaled_qs, 
                                        param_type='charge')
    sigs = make_precomputed_param_vectors(all_par[:,1], 
                                        scaled_indices=scaled_indices,
                                        scaled_params=scaled_sigs, 
                                        param_type='sigma')
    eps = make_precomputed_param_vectors(all_par[:,2], 
                                        scaled_indices=scaled_indices,
                                        scaled_params=scaled_epsilons, 
                                        param_type='epsilon')
    
    

    intra_res_indices = get_intra_res_indices(resis, atoms)

    mask_indices = np.array(list(set(intra_res_indices + exclusion_indices)))

    energies = get_trajectory_interaction_energies(u, qs, sigs, eps, sel_string)
    res_energies = get_pairwise_residue_energies(energies, mask_indices, resis, atoms)

    # save the atom, mask, and energy arrays
    

