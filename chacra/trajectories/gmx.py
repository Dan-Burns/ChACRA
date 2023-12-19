import MDAnalysis as mda
from MDAnalysis.analysis import align
import parmed as pmd
import subprocess
import os
import glob

#TODO - test
# Function to concatenate and process trajectories
def process_trajectories(folder, 
                         selection='not (resname HOH or resname CL or resname NA)',
                          index=None, out_index=None, 
                          traj_ext='xtc', out_traj_ext=None, 
                          align_structure=None):
    '''
    Concatenates all the {traj_ext} files in folder and then performs 2 gmx 
    trjconv steps of periodic boundary condition fixes before running a final 
    alignment step.  

    selection : str
        An MDAnalysis selection string specifying the contents of the outputted 
        trajectory. The selection and gromacs index file indices must contain
        compatible system contents. If you just want the protein (and ligand) 
        contents with no solvent then make sure that the selection removes 
        everything except these groups from the input trajectory and the gromacs
          index corresponds to just the protein (and ligand).

    folder : str
        The name of the folder containing the trajectory files that are to be 
        processed.  Anything with traj_ext will get concatenated so your folder
        should not contain equilibration or other trajectories.  Naming scheme 
        should be such that the last characters before the file extension are
        digits indicating the run number. If your replica exchange trajectories 
        were produced with gromacs, this should already be the case.

    index : int
        The gromacs index file index corresponding the group that should be 
        centered and utilized for treating periodic boundary crossings.

    out_index : int or None
        The gromacs index file index corresponding to the system contents that 
        should be ouputted.1 corresponds to "protein" by default. If you specify
          "1",then "selection" should be "protein".

    traj_ext : str
        The file extension for the original trajectories. Default is "xtc".

    out_traj_ext : str
        The desired file extension for the final output trajectory.
        Defaults to traj_ext.

    align_structure : str 
        Path to a structure file to use as the reference for final alignment 
        step.  Contents must match the selections. If None, align to first frame 
        in the trajectory.
    
        
    Example
    -------
    # Main processing loop
    # 20 replica directions name tpr_00 - tpr_19
    # gromacs index 14 corresponds to whatever the desired outputs components 
    # are here (protein + ligand)
    # index 1 corresponds to "protein" by default
    for i in range(20):
        folder_name = f"tpr_{i:02d}"
        print(f"Processing folder: {folder_name}")
        process_trajectories(folder_name, index=14, 
        align_structure='protein_nohoh.pdb')

        print(f"Finished processing folder: {folder_name}")

    print("All trajectories processed.")
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
    trjconv_input1 = f'printf "%s\\n" "{index}" "{out_index}" | gmx trjconv '\
        f'-s {topol_file} -f intermediate_step.{traj_ext} '\
        f'-o intermediate_step2.{traj_ext} -center -pbc nojump -n index.ndx'
    subprocess.run(trjconv_input1, shell=True)

    # Remove the first intermediate file
    #os.remove(os.path.join(folder, f'intermediate_step.{traj_ext}'))
    os.remove(f'intermediate_step.{traj_ext}')

    # Second gmx trjconv call
    # output_file = os.path.join('processed_trajs', 
    #                            f'centered_replica_{folder[-2:]}.{traj_ext}')
    trjconv_input2 = f'printf "%s\\n" "{index}" "{index}" "{out_index}" | '\
        f'gmx trjconv -s {topol_file} -f intermediate_step2.{traj_ext} '\
        f'-o intermediate_step3.{traj_ext} -pbc cluster -n index.ndx -center'
    subprocess.run(trjconv_input2, shell=True)

    # Remove the second intermediate file
    os.remove(f'intermediate_step2.{traj_ext}')

    # final mdanalysis alignment
    if align_structure is None:
        align_structure = topol_file
        ref = mda.Universe(align_structure)
        sel = ref.select_atoms(selection)
        ref = mda.Merge(sel)
        u = ref.copy() 
        u.load_new(f'intermediate_step3.{traj_ext}')
    
    else:
        ref = mda.Universe(align_structure)
        u = mda.Universe(align_structure, f'intermediate_step3.{traj_ext}')
    

    align.AlignTraj(u,
                    ref,
                    select="name CA",
                    filename=f"processed_trajs/rep_{folder[-2:]}.{out_traj_ext}",
                ).run()
    os.remove(f'intermediate_step3.{traj_ext}')




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