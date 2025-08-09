![chacra_logo](https://github.com/Dan-Burns/ChACRA/assets/58605062/a030ffbb-0a97-4b33-a968-fab2ec7dbee9)

# ChACRA


## **Ch**emically **A**ccurate **C**ontact **R**esponse **A**nalysis

Created by Dan Burns
https://github.com/Dan-Burns/ChACRA


Tools for identifying energy-sensitive interactions in proteins using contact data from replica exchange molecular dynamics simulations (REMD). The energy-sensitive interaction modes (or chacras) are the principal components of a protein's contact frequencies across temperature. The chacras reveal functionally critical residue interactions through the highest loading score contacts. Allosteric communication is suggested when distinct parts of the structure are characterized by the same chacra.

With ChACRA you can run the full pipeline including the replica exchange simulations and contact calculations with a single command. 

### Installation

Clone and enter the repository.

```
git clone --recurse-submodules https://github.com/Dan-Burns/ChACRA.git && cd ChACRA
```

Create the conda environment. It's recommended to use [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
```
micromamba env create -f environment.yaml 
```

```
micromamba activate chacra-env
```

Then install ChACRA.
```
pip install -e .
```

### Run the example

Create and enter the example directory.

```
mkdir ~/chacra_example && cd ~/chacra_example
```

Setup the working directory.

```
chacra-project --example
```
The "--example" flag just copies the 1tnf.pdb into structures/. 

Solvate the structure and create an [OpenMM](https://github.com/openmm) system to simulate with replica exchange.
```
make-simulation -s structures/1tnf.pdb --fix --name 1tnf_example
```
The "--fix" flag will use OpenMM's pdbfixer to automatically protonate the structure and can insert missing residues if a .cif file is provided with the full sequence. Always check the output structure. Missing residues are placed naively and can make the termini extend out, creating a overly large simulation box. You'll see that a 1tnf_example_minimized.pdb is in the structures/ directory and 1tnf_example_system.xml is in the system/ directory.

Now you can run Hamiltonian replica exchange molecular dynamics (HREMD) which by default will apply the Hamiltonian scaling to all the protein atoms. HREMD is implemented with [femto](https://github.com/Psivant/femto).

```
run-hremd --system_file system/1tnf_example_system.xml \
          --structure_file structures/1tnf_example_minimized.pdb \
          --n_cycles 1000
          --j 16
          --n 16          
```
This command will run 1000 replica exchange cycles with 1000 timesteps per cycle (default), saving coordinates every 10 cycles (default). You can add warmup steps before the replica exchange begins to allow for equilibration and decorrelation of the systems at the different Hamiltonian scalings. "run-hremd --help" details the available options.

16 replicas were specified for the small example system. For systems with 100,000 to 400,000 particles you might need anywhere from 20-40 replicas to obtain adequate exchange probabilities.

If you encounter errors here it could be due to starting coordinates that aren't adequately minimized or equilibrated. 
Another common error is related to CUDA driver version incompatibility with OpenMM dependencies.

run-hremd also calls process-output to automatically generate the state trajectories, run the contact calculations, and write some ChACRA output. These outputs are found in state_trajectories/run_{i}, contact_output/run_{i}, and analysis_output/run_{i}. 

A .pml file and a .csv is write to the analysis_output/run_{i} directory so you can visualize the chacras and know which contacts are most sensitive on each chacra. The total_contacts.pd is the accumulated contact frequency data for all the runs and the .pml and .csv file reflects all of the combined runs as well. You should keep running until these outputs converge. 

The output will report on any chacra (principal component) that passes a significance test. The energy-dependent response patterns (pc projections) can be seen with the chacra_modes.png plot. 

![chacras](https://github.com/Dan-Burns/ChACRA/assets/58605062/00a98056-bd79-4a3f-95ec-656688838301)

Drop your pdb file and the .pml file into PyMol to see the most sensitive contacts on the structure. They will be colored according to the response pattern they exhibit.

![IGPS_chacras](https://github.com/Dan-Burns/ChACRA/assets/58605062/a8eb2448-26e5-48e6-a421-6b4cc798ac33)

Further, the example structure is a homotrimer and the contact data can be averaged to make the results more statistically robust and easier to visualize.

An interactive analysis notebook is available in examples/example_notebook.ipynb that should make the ChACRA output clear.


1. Burns, D., Singh, A., Venditti, V. & Potoyan, D. A. Temperature-sensitive contacts in disordered loops tune enzyme I activity. Proc. Natl. Acad. Sci. U. S. A. 119, e2210537119 (2022)

2. Burns, D., Venditti, V. & Potoyan, D. A. Temperature sensitive contact modes allosterically gate TRPV3. PLoS Comput. Biol. 19, e1011545 (2023)

3. Burns, D., Venditti, V. &#38; Potoyan, D. A. Illuminating protein allostery by chemically accurate contact response analysis (ChACRA). <i>J. Chem. Theory Comput.</i> (2024)

