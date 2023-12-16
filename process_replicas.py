import MDAnalysis as mda
from MDAnalysis.analysis import align
import parmed as pmd
import subprocess
import os
import glob

# Function to concatenate and process trajectories
def process_trajectories(folder, 
                         selection='not (resname HOH or resname CL or resname NA)',
                          index=None, out_index=None, 
                          traj_ext='xtc', out_traj_ext=None, 
                          align_structure=None):
    '''
    Concatenates all the {traj_ext} files in folder and then performs 2 gmx trjconv steps of periodic boundary condition fixes before running
    a final alignment step.  

    selection : string
        An MDAnalysis selection string specifying the contents of the outputted trajectory. The selection and gromacs index file indices must contain
        compatible system contents. If you just want the protein (and ligand) contents with no solvent then make sure that the selection removes everything
        except these groups from the input trajectory and the gromacs index corresponds to just the protein (and ligand).

    folder : string
        The name of the folder containing the trajectory files that are to be processed.  Anything with traj_ext will get concatenated so 
        your folder should not contain equilibration or other trajectories.  Naming scheme should be such that the last characters before the file
        extension are digits indicating the run number. If you replica exchange trajectories were produced with gromacs, this should already be the case.

    index : int
        The gromacs index file index corresponding the group that should be centered and utilized for treating periodic boundary crossings.

    out_index : int or None
        The gromacs index file index corresponding to the system contents that should be ouputted.

    traj_ext : str
        The file extension for the original trajectories

    out_traj_ext : str
        The desired file extension for the final output trajectory. Defaults to traj_ext.

    align_structure : str 
        Path to a structure file to use as the reference for final alignment step.  Contents must match the selections. If None,
        align to first frame in the trajectory.
    
    '''

    topol_file = os.path.join(folder, 'topol.tpr')
    traj_files = sorted(glob.glob(os.path.join(folder, f'*.{traj_ext}')))

    if out_traj_ext == None:
        out_traj_ext = traj_ext

    if out_index == None:
        out_index = index

    # Create a MDAnalysis universe
    u = mda.Universe(topol_file, traj_files)

    # Write the concatenated trajectory
    if selection is None:
        selection = 'all'
    selection = u.select_atoms(selection)
    selection.write(f'intermediate_step.{traj_ext}', frames='all')


    # First gmx trjconv call
    trjconv_input1 = f'printf "%s\\n" "{index}" "{out_index}" | gmx trjconv -s {topol_file} -f intermediate_step.{traj_ext} -o intermediate_step2.{traj_ext} -center -pbc nojump -n index.ndx'
    subprocess.run(trjconv_input1, shell=True)

    # Remove the first intermediate file
    #os.remove(os.path.join(folder, f'intermediate_step.{traj_ext}'))
    os.remove(f'intermediate_step.{traj_ext}')

    # Second gmx trjconv call
    output_file = os.path.join('processed_trajs', f'centered_replica_{folder[-2:]}.{traj_ext}')
    trjconv_input2 = f'printf "%s\\n" "{index}" "{index}" "{out_index}" | gmx trjconv -s {topol_file} -f intermediate_step2.{traj_ext} -o intermediate_step3.{traj_ext} -pbc cluster -n index.ndx -center'
    subprocess.run(trjconv_input2, shell=True)

    # Remove the second intermediate file
    os.remove(f'intermediate_step2.{traj_ext}')

    # final mdanalysis alignment
    if align_structure is None:
        align_structure = topol_file
    u = mda.Universe(align_structure, f'intermediate_step3.{traj_ext}')
    ref = mda.Universe(align_structure)
    align.AlignTraj(u,
                    ref,
                    select="name CA",
                    filename=f"processed_trajs/rep_{folder[-2:]}.{out_traj_ext}",
                ).run()
    os.remove(f'intermediate_step3.{traj_ext}')


# Main processing loop
for i in range(2):
    folder_name = f"tpr_{i:02d}"
    print(f"Processing folder: {folder_name}")
    process_trajectories(folder_name, index=14, align_structure='1gpw_prfar_nohoh.pdb')

    print(f"Finished processing folder: {folder_name}")

print("All trajectories processed.")

def parmed_underscore_topology(gromacs_processed_top, atom_indices, output_top):
    '''
    Add underscores to atom types of selected atoms.
    This is useful if using the plumed_scaled_topologies script 
    for hremd system modification.
    With this, you still need to open the new topology file and delete the 
    underscores from the beginning of the file [atomtypes]
    or else plumed will look for atoms with 2 underscores to apply lambda to.
    '''
    top = GromacsTopologyFile(gromacs_processed_top)

    for atom in top.view[atom_indices].atoms:
        atom.type = f"{atom.type}_"
        if atom.atom_type is not pmd.UnassignedAtomType:
            atom.atom_type = copy.deepcopy(atom.atom_type)
            atom.atom_type.name = f"{atom.atom_type.name}_"


    top.save(output_top)

#TODO example plumed/ replica gmx grompp
    







########### PROLIF SANDBOX #################
# results with these parameters are quite different than getcontacts results
# at least 1/3 of the contacts differ by more than 10% - calculated at stride 10 so unless the initial
# frame was different, only difference can be the cutoffs and angles are still not right
# OR atom classification from prolif SMARTS pattern is picking up different atoms.
# prolif also recorded about 10% more interactions.
    
# Something to try. Subclass VdWcontact for hydrophobic and make sure the right atoms are in it
# and then use the same tolerance -
    

get_contacts_parameters = {
            
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

            'Hydrophobic':{"distance":4,}, # Needs to be a 0.5 threshold + VdW radii
                                
            'HBAcceptor':{
                        "DHA_angle":(110,180)}, #<70 from AHD
            
            'HBDonor':{
                        "DHA_angle":(110,180)},
               
            'PiCation':{"distance":6.0, 
                        "angle":(0,60)},
        
            'VdWContact':{"tolerance":0.5, 
                        },
            }

class HydrophobicMod(plf.interactions.VdWContact):
    def __init__(self, tolerance=0.5, vdwradii=None):
        # Initialize the parent class with the specified tolerance and vdwradii
        super().__init__(tolerance=tolerance, vdwradii=vdwradii)
        # Define the hydrophobic SMARTS pattern
        hydrophobic = (
            "[c,s,Br,I,S&H0&v2,$([D3,D4;#6])&!$([#6]~[#7,#8,#9])&!$([#6X4H0]);+0]"
        )

    def detect(self, ligand, residue):
        # Filter atoms in ligand and residue based on the hydrophobic pattern
        hydrophobic_ligand_atoms = [
            atom for atom in ligand.GetAtoms() if atom.HasSubstructMatch(self.hydrophobic_pattern)
        ]
        hydrophobic_residue_atoms = [
            atom for atom in residue.GetAtoms() if atom.HasSubstructMatch(self.hydrophobic_pattern)
        ]

        # Proceed with the original VdWContact detection logic, but only for hydrophobic atoms
        lxyz = ligand.GetConformer()
        rxyz = residue.GetConformer()
        for la, ra in product(hydrophobic_ligand_atoms, hydrophobic_residue_atoms):
            lig = la.GetSymbol()
            res = ra.GetSymbol()
            elements = frozenset((lig, res))
            try:
                vdw = self._vdw_cache[elements]
            except KeyError:
                vdw = self.vdwradii.get(lig, VDWRADII[lig]) + self.vdwradii.get(res, VDWRADII[res]) + self.tolerance
                self._vdw_cache[elements] = vdw
            dist = lxyz.GetAtomPosition(la.GetIdx()).Distance(
                rxyz.GetAtomPosition(ra.GetIdx())
            )
            if dist <= vdw:
                yield self.metadata(
                    ligand, residue, (la.GetIdx(),), (ra.GetIdx(),), distance=dist
                )
    
def get_prolif_contacts(topology, trajectory, guess_bonds=False, sel1='protein',sel2='protein',itypes='all',
                        contacts_output='contacts.pd', contact_frequencies_output='contact_frequencies.pd'):
    '''
    Not implemented

    
    
    '''
    structure = '../gromacs_1gpw_prfar/1gpw_prfar_slow_heat.tpr'
    u = mda.Universe(structure)

    selection = u.select_atoms('not (resname HOH or resname NA or resname CL)')
    traj = '../from_nova/rep_00.xtc'
    u = mda.Merge(selection)
    u.load_new(traj)
    import prolif as plf
    import MDAnalysis as mda
    import pandas as pd
    import numpy as np

    #structure = '1gpw_prfar_slow_heat.tpr'
    structure = '1gpw_prfar_nohoh.pdb'
    u = mda.Universe(structure)

    #selection = u.select_atoms('not (resname HOH or resname NA or resname CL)')
    #traj = 'processed_trajs/rep_00.xtc'
    #u = mda.Merge(selection)
    #u.load_new(traj)

    protein = u.select_atoms('protein')
    protein.guess_bonds()

    fp = plf.Fingerprint(['Anionic',
    'CationPi',
    'Cationic',
    'EdgeToFace',
    'FaceToFace',
    'HBAcceptor',
    'HBDonor',
    'Hydrophobic',
    'PiCation',
    'PiStacking',
    'VdWContact',
    ])
    fp.run(u.trajectory[::10], protein, protein, n_jobs=36)

    df = fp.to_dataframe()

    df.to_pickle('traj_0_prolif_bonds.pd')

def get_prolif_freqs():
    '''
    Not Implemented
    '''
    reslabels = set([a for tup in df.columns for a in tup[:1]])
    data = {}
    n_frames = df.shape[0]
    for res in reslabels:
        res1 = mapper[res] # added to adjust for mismatch from using tpr
        resa, chaina = res1.split(".")# changed
        resna, resida = resa[:3], resa[3:]
        cols = set([a for tup in df.xs(res,level="ligand",axis=1).columns for a in tup[:1]])
        for col in cols:
            col2 = mapper[col] # added to adjust
            resb, chainb = col2.split(".")# changed
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
                #data[contact] = bool_contacts.index[bool_contacts.values.squeeze()].shape[0]/n_frames
                result = len(bool_contacts.loc[bool_contacts.any(axis=1)==True])/n_frames
                if np.allclose(result, 0.0):
                    pass
                else:
                    data[contact] = result