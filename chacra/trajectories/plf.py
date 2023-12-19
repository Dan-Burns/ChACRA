import prolif as plf
import numpy as np
import pandas as pd

# These parameters should match the default from getcontacts.
# However the results with these parameters are still different - could be SMARTS patterns...
# https://getcontacts.github.io/interactions.html
getcontacts_params = {
            
            'Anionic':{"distance":4.0, # considered 4.5 by prolif
                       },
 
            'CationPi':{"distance":6.0, # 4.5 for prolif
                        "angle":(0,60)},# 0 to 30 for prolif
            
            'Cationic':{"distance":4.0,
                        },
      
            'EdgeToFace':{"distance":5.0, 
                        "plane_angle":(60,90), # (50, 90) in prolif
                        "normal_to_centroid_angle":(0,45)}, # (0,30) in prolif
            
            'FaceToFace':{"distance":7.0, # 5.5 for prolif
                        "plane_angle":(0,30),
                        "normal_to_centroid_angle":(0,45)},

            'HydrophobicVdW':{"threshold":0.5,}, # Original 'Hydrophobic' is "distance":4.0
                                
            'HBAcceptor':{
                        "DHA_angle":(110,180)}, #<70 from AHD
            
            'HBDonor':{
                        "DHA_angle":(110,180)},
               
            'PiCation':{"distance":6.0, 
                        "angle":(0,60)},
        
            'VdWContact':{"tolerance":0.5, 
                        },
            }

from itertools import product
from prolif.interactions import Hydrophobic, VdWContact
class HydrophobicVdW(Hydrophobic, VdWContact):
    '''
    Courtesy of cbouy (ProLIF author) via github issues.

    Usage
    -----
    from ChACRA.chacra.(appropriate_module) import HydrophobicVdW

    fp = plf.Fingerprinte(parameters=getcontacts_params, interactions=["HydrophobicVdW"])
    fp.run(u.trajectory, ligand_sel, protein_sel)
    '''
    def __init__(self, tolerance=0.5):
        Hydrophobic.__init__(self)
        VdWContact.__init__(self, tolerance=tolerance)

    def detect(self, lig_res, prot_res):
        # match hydrophobic SMARTS
        lig_matches = lig_res.GetSubstructMatches(self.lig_pattern)
        prot_matches = prot_res.GetSubstructMatches(self.prot_pattern)
        if lig_matches and prot_matches:
            # satisfy VdW radii sum cutoff
            lxyz = lig_res.GetConformer()
            rxyz = prot_res.GetConformer()
            for lig_match, prot_match in product(lig_matches, prot_matches):
                la = lig_res.GetAtomWithIdx(lig_match[0])
                ra = prot_res.GetAtomWithIdx(prot_match[0])
                lig = la.GetSymbol()
                res = ra.GetSymbol()
                elements = frozenset((lig, res))
                try:
                    vdw = self._vdw_cache[elements]
                except KeyError:
                    vdw = self.vdwradii[lig] + self.vdwradii[res] + self.tolerance
                    self._vdw_cache[elements] = vdw
                dist = lxyz.GetAtomPosition(lig_match[0]).Distance(
                    rxyz.GetAtomPosition(prot_match[0])
                )
                if dist <= vdw:
                    yield self.metadata(
                        lig_res, prot_res, lig_match, prot_match, distance=dist
                    )
    
def get_prolif_contacts(topology, 
                        trajectory, 
                        selection='not (resname HOH or resname NA or resname CL)',
                        guess_bonds=False, 
                        sel1='protein',
                        sel2='protein',
                        stride=1,
                        n_jobs=None,
                        contacts_output='prolif_contacts.pd', 
                        contact_frequencies_output='prolif_contact_frequencies.pd',
                        params=getcontacts_params,
                        contact_types=['Anionic',
                        'CationPi',
                        'Cationic',
                        'EdgeToFace',
                        'FaceToFace',
                        'HBAcceptor',
                        'HBDonor',
                        'HydrophobicVdW',
                        'PiCation',
                        'PiStacking',
                        'VdWContact',]
                        ):
    '''
    Not implemented

    
    
    '''
    structure = topology
    u = mda.Universe(structure)

    selection = u.select_atoms(selection)
    traj = trajectory
    u = mda.Merge(selection)
    u.load_new(traj)

    # only allowing for protein protein contacts with this...
    sel1 = u.select_atoms(sel1)
    sel2 = u.select_atoms(sel2)
    if guess_bonds == True:
        sel1.guess_bonds()
        sel2.guess_bonds()

    fp = plf.Fingerprint(parameters=params,interactions=contact_types)
    fp.run(u.trajectory[::stride], sel1, sel2, n_jobs=n_jobs)

    df = fp.to_dataframe()

    df.to_pickle(contacts_output)

    freqs = get_prolif_freqs(df)
    freqs.to_pickle(contact_frequencies_output)
    

def get_prolif_freqs(df):
    '''
    Not Implemented
    Gets the contact frequencies where one or more contact happening during a frame counts as 1 contact
    for that frame.  

    If you need to map from a tpr input back to a pdb with chain ids/ different numbering
    u=mda.Universe('protein.pdb')
    protein = u.select_atoms('protein')
    u_tpr=mda.Universe('topol.tpr')
    protein_tpr = u_tpr.select_atoms('protein')
    mapper = {}
    for res_tpr, res in zip(protein_tpr.residues, protein.residues):
        tpr_name = f'{res_tpr.resname}{res_tpr.resid}'
        name = f'{res_tpr.resname}{res.resid}.{res.segid}'
        mapper[tpr_name]=name

    TODO: provide mapper options for case where topology doesn't match pdb/
     or desired resids
    TODO: option to base frequency for certain residues off of specific
    contact type - e.g. hydrophobic
    '''
    reslabels = set([a for tup in df.columns for a in tup[:1]])
    data = {}
    n_frames = df.shape[0]
    for res in reslabels:
        #res1 = mapper[res] # added to adjust for mismatch from using tpr
        resa, chaina = res.split(".") # change to res1 for mapper option
        resna, resida = resa[:3], resa[3:]
        cols = set([a for tup in df.xs(res,level="ligand",axis=1).columns for a in tup[:1]])
        for col in cols:
            #col2 = mapper[col] # added to adjust
            resb, chainb = col.split(".") # change to col2 for mapper
            resnb, residb = resb[:3], resb[3:]
            if f'{chaina}:{resna}:{resida}' == f'{chainb}:{resnb}:{residb}':
                continue
            elif chaina < chainb:
                contact = f'{chaina}:{resna}:{resida}-{chainb}:{resnb}:{residb}'
            elif chaina > chainb:
                contact = f'{chainb}:{resnb}:{residb}-{chaina}:{resna}:{resida}'                  
            elif chaina == chainb and resna < resnb:
                contact = f'{chaina}:{resna}:{resida}-{chainb}:{resnb}:{residb}'                    
            elif chaina == chainb and resna > resnb:
                contact = f'{chainb}:{resnb}:{residb}-{chaina}:{resna}:{resida}'
            elif chaina == chainb and resna == resnb and resida < residb:
                contact = f'{chaina}:{resna}:{resida}-{chainb}:{resnb}:{residb}' 
            elif chaina == chainb and resna == resnb and resida > residb:
                contact = f'{chainb}:{resnb}:{residb}-{chaina}:{resna}:{resida}'
            else:
                print(f'{chaina}:{resna}:{resida}-{chainb}:{resnb}:{residb}')
                                
            if contact in data.keys():
                continue
            else:
                
                bool_contacts = df.xs(res,level="ligand",axis=1).xs(col, level='protein', axis=1)
                
                result = len(bool_contacts.loc[bool_contacts.any(axis=1)==True])/n_frames
                if np.allclose(result, 0.0):
                    pass
                else:
                    data[contact] = result
    return pd.DataFrame(data, index=[0])