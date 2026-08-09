![chacra_logo](https://github.com/Dan-Burns/ChACRA/assets/58605062/a030ffbb-0a97-4b33-a968-fab2ec7dbee9)

# ChACRA


## **Ch**emically **A**ccurate **C**ontact **R**esponse **A**nalysis

Created by Dan Burns
https://github.com/Dan-Burns/ChACRA


Tools for identifying energy-sensitive interactions in proteins using contact data from replica exchange molecular dynamics simulations (REMD). The energy-sensitive interaction modes (or chacras) are the principal components of a protein's contact frequencies across temperature. The chacras reveal functionally critical residue interactions through the highest loading score contacts. Allosteric communication is suggested when distinct parts of the structure are characterized by the same chacra.

With ChACRA you can run the full pipeline including the replica exchange simulations and contact calculations with a single command. 

### Installation

To run ChACRA's replica exchange simulations across multiple GPUs, you must have OpenMPI and NVIDIA drivers installed on your system.

Clone and enter the repository.

```bash
git clone https://github.com/Dan-Burns/ChACRA.git
cd ChACRA
```

Run the automated installation script. This script will:
1. Verify system prerequisites (`mpicc`, `nvidia-smi`).
2. Detect your CUDA version and determine the correct `cupy` package.
3. Create the conda environment (`chacra-env`).
4. Install `mpi4py` built against your system MPI.
5. Install the `femto`, `ultracontacts`, and `getcontacts` main branches via pip.

```bash
./install.sh
```

Then activate the environment:
```bash
conda activate chacra-env
```
### Usage
#### Run an example

Create and enter the example directory.

```
mkdir ~/chacra_example && cd ~/chacra_example
```

Setup the working directory.

```
chacra project --example
```
The "--example" flag just copies the 1tnf_truncated.pdb into structures/. 

Solvate the structure and create an [OpenMM](https://github.com/openmm) system to simulate with replica exchange.
```
chacra make-simulation -s structures/1tnf_truncated.pdb --fix --name 1tnf_example
```
The "--fix" flag will use OpenMM's pdbfixer to automatically protonate the structure and can insert missing residues if a .cif file is provided with the full sequence. Always check the output structure. Missing residues are placed naively and can make the termini extend out, creating a overly large simulation box. You'll see that a 1tnf_example_minimized.pdb is in the structures/ directory and 1tnf_example_system.xml is in the system/ directory.

Now you can run Hamiltonian replica exchange molecular dynamics (HREMD) which by default will apply the Hamiltonian scaling to all the protein atoms. HREMD is implemented with [femto](https://github.com/Psivant/femto). femto will spread the systems out between the available GPUs on the node. Assuming a node with 4 GPUs and 20 replicas...

```
chacra run-hremd --system_file system/1tnf_example_system.xml \
          --structure_file structures/1tnf_example_minimized.pdb \
          --n_cycles 1000 \
          -j 4 \
          -n 20          
```
This command will run 1000 replica exchange cycles with 1000 timesteps per cycle (default), saving coordinates every 10 cycles (default). You can add warmup steps before the replica exchange begins to allow for equilibration and decorrelation of the systems at the different Hamiltonian scalings. 

#### Restarts
To continue running, just execute the above command again and a new run/ folder will be created in each of the directories. You'll find the extended run output there when the script exits.

#### Output
`chacra run-hremd` also calls `chacra process-output` to automatically generate the state trajectories, run the contact calculations, and write some ChACRA output. These outputs are found in state_trajectories/run_{i}, contact_output/run_{i}, and analysis_output/run_{i}. If `process-output` fails partway through, you can rerun it directly — it will skip completed stages and resume from where it left off.

A .pml file and a .csv is written to the analysis_output/run_{i} directory so you can visualize the chacras and know which contacts are most sensitive on each chacra. The total_contacts.pd pandas dataframe reflects the accumulated data for all the runs and the .pml and .csv file reflects all of the combined runs as well. You should keep running until these outputs converge. The .csv file provides the names of the most sensitive interactions on each chacra. The residues in the first couple contacts in each column can be good targets for structure-activity investigations.

The output will report on any chacra (principal component) that passes a significance test. The energy-dependent response patterns (pc projections) can be seen with the chacra_modes.png plot. You can choose to examine fewer chacras than the significance test suggests by running your own analysis in a notebook (see the analysis.ipynb in the examples directory). If you run the simulations long enough, the significance test should identify fewer significant chacras and their projection plots should become smooth.

#### Visualization

![chacras](https://github.com/Dan-Burns/ChACRA/assets/58605062/00a98056-bd79-4a3f-95ec-656688838301)

*Figure 1. Projections of the contact frequency principal components (chacras). You can see how the red mode (1st chacra / PC1) captures a melting trend of decreasing probability with increasing temperature. The blue mode (2nd chacra) captures a trend of increasing contact probaility with increasing temperature. This tendency towards increased order at increased energy and at the expense of the increasing disorder of the 1st chacra often reveals functionaly critical interactions among residues involved in the highest loading score contacts on PC2.*

Drop your pdb file and the .pml file into PyMol to see the most sensitive contacts on the structure. They will be colored according to the response pattern they exhibit.

![IGPS_chacras](https://github.com/Dan-Burns/ChACRA/assets/58605062/a8eb2448-26e5-48e6-a421-6b4cc798ac33)

*Figure 2. The most sensitive interactions on the chacras of the allosterically activated enzyme IGPS [3] (this is not the example protein). The fifth chacra (orange) captures the allosterically coupled active site and effector binding site. The second chacra (blue) captures interactions critical for activity.*

Further, the example structure is a homotrimer and the contact data can be averaged to make the results more statistically robust and easier to visualize. An interactive analysis notebook is available in examples/example_notebook.ipynb that demonstrates this.

#### Notes
20 replicas were used in the example which results in a 10-15% exchange rate. More frequent exchanges require more systems. For systems with 50,000 to 300,000 particles you might need anywhere from 20-40 replicas to obtain adequate exchange probabilities. Determining the number of systems to return an exchange rate of ~15-25% is a trial and error process. 

If you set up a system with a ligand, you may not be able to run femto HREMD without editing the femto source code. It's pretty simple though. You just need to add the name of any custom force object that the ligand parameters require to femto/md/rest.py's _SUPPORTED_FORCES list. The HREMD scaling will be limited to the protein though.

#### Common Errors
If you encounter particle NaN errors running HREMD, it's likely due to starting coordinates that aren't adequately energy minimized or equilibrated. 

Another common error is `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)`, which means the CUDA toolkit packages in the conda environment are newer than what your GPU driver supports. The `install.sh` script handles this automatically by setting `CONDA_OVERRIDE_CUDA` to match your driver. If you installed manually, recreate the environment with:
```bash
CONDA_OVERRIDE_CUDA=$(nvidia-smi | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+') \
  conda env update -f environment.yaml --prune
```


#### Citations
Please cite the following if you use ChACRA.

1. Burns, D., Singh, A., Venditti, V. & Potoyan, D. A. Temperature-sensitive contacts in disordered loops tune enzyme I activity. Proc. Natl. Acad. Sci. U. S. A. 119, e2210537119 (2022)

2. Burns, D., Venditti, V. & Potoyan, D. A. Temperature sensitive contact modes allosterically gate TRPV3. PLoS Comput. Biol. 19, e1011545 (2023)

3. Burns, D., Venditti, V. &#38; Potoyan, D. A. Illuminating protein allostery by chemically accurate contact response analysis (ChACRA). <i>J. Chem. Theory Comput.</i> (2024)

