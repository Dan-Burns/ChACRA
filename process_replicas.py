import MDAnalysis as mda
from MDAnalysis.analysis import align
import subprocess
import os
import glob

# Function to concatenate and process trajectories
def process_trajectories(folder, index=None, align=True, align_structure=None):
    topol_file = os.path.join(folder, 'topol.tpr')
    traj_files = sorted(glob.glob(os.path.join(folder, '*.xtc')))


    # Create a MDAnalysis universe
    u = mda.Universe(topol_file, traj_files)

    # Write the concatenated trajectory
    selection = u.select_atoms('not (resname HOH or resname CL or resname NA)')
    selection.write('intermediate_step.xtc', frames='all')


    # First gmx trjconv call
    trjconv_input1 = f'printf "%s\\n" "{index}" "{index}" | gmx trjconv -s {topol_file} -f intermediate_step.xtc -o intermediate_step2.xtc -center -pbc nojump -n index.ndx'
    subprocess.run(trjconv_input1, shell=True)

    # Remove the first intermediate file
    #os.remove(os.path.join(folder, 'intermediate_step.xtc'))
    os.remove('intermediate_step.xtc')

    # Second gmx trjconv call
    output_file = os.path.join('processed_trajs', f'centered_replica_{folder[-2:]}.xtc')
    trjconv_input2 = f'printf "%s\\n" "{index}" "{index}" "{index}" | gmx trjconv -s {topol_file} -f intermediate_step2.xtc -o intermediate_step3.xtc -pbc cluster -n index.ndx -center'
    subprocess.run(trjconv_input2, shell=True)

    # Remove the second intermediate file
    os.remove('intermediate_step2.xtc')

    # final mdanalysis alignment
    u = mda.Universe(align_structure, 'intermediate_step3.xtc')
    ref = mda.Universe(align_structure)
    align.AlignTraj(u,
                    ref,
                    select="name CA",
                    filename=f"processed_trajs/rep_{folder[-2:]}.xtc",
                ).run()
    os.remove('intermediate_step3.xtc')
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