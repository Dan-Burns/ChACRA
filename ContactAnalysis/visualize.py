import nglview as nv
import MDAnalysis as mda
from colors import chacra_colors, hex_to_RGB

# class to easily depict ChACRA data in nglview
# if you want more than one atom at a time use np.isin
#np.where(np.isin(u.atoms.resids,[2,1]) & (u.atoms.segids == 'G') & (u.atoms.names == 'CA'))

# offer in pointing or away pointing arrows depending on contact breaking, forming
# offer cylinder width option to depict most sensitive contact with largest cylinder

def get_midpoint(a,b):
    '''
    Get the midpoint of two vectors.

    Parameters
    ----------
    a,b : np.array
    Returns
    -------
    np.array 
    '''
    return np.mean([a,b], axis=0)

def get_contact_indices(contact, ca_only=True):
    '''
    Get the atom or residue indices from an mda.Universe/ structure

    Parameters 
    ----------
    contact : string
        Contact name. 
    
    ca_only : bool
        Return the atom indices of the c-alpha atoms.
        Default is True. If you want the residue index, set ca_only=False

    Returns
    np.array of atom or residue indices
    '''
    indices = []
    for res in contact:
        if ca_only:
            indices.append(np.where((u.atoms.resids == resid) & (u.atoms.segids == segid) & (u.atoms.names == 'CA')))
        else:
            indices.append(np.where((u.resids == resid) & (u.residues.segids == segid))) 
    return np.asarray(indices)

def get_positions(atom_indices):
    '''
    
    '''

def draw_line(contact, arrows=False, width=0.3, variable_width=False):
    '''
    
    '''

    v.view.shape.add_arrow(positions[1],midpoint(positions[0],positions[1]),[1,0,0],0.8)

class Visualizer:
    '''
    structure : str or mda.Universe
        Path to structure file or mda.Universe corresponding to the 
        
    cpca : ContactPCA

    cont : ContactFrequencies
    
    '''
    def __init__(self, structure, cpca, cont):
        if type(structure) == mda.core.universe.Universe:
            self.u = structure
        else:
            self.u = mda.Universe(structure)

        self.view = nv.show_mdanalysis(self.u)
        self.cpca = cpca
        self.cont = cont

    def view(self):
        return self.view
    
    def show_chacra(self, pc, clear_representation=False):
