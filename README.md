![chacra_logo](https://github.com/Dan-Burns/ChACRA/assets/58605062/a030ffbb-0a97-4b33-a968-fab2ec7dbee9)

# ChACRA


## **Ch**emically **A**ccurate **C**ontact **R**esponse **A**nalysis

Created by Dan Burns
https://github.com/Dan-Burns/ChACRA


Tools for identifying energy-sensitive interactions in proteins using contact data from replica exchange molecular dynamics simulations (REMD).  The energy-sensitive interaction modes (or chacras) are the principal components of a protein's contact frequencies across temperature.  The chacras can reveal functionally critical residue interactions.

To start, first generate a set of md trajectories across a temperature range.  This is should done using the Hamiltonian REMD method also known as Replica Exchange with Solute Tempering (REST2)[1] .  The [plumed implementation](https://www.plumed.org/doc-v2.9/user-doc/html/hrex.html) is a good way of going about this. 

Once you have your trajectories, generate your contact data using [getcontacts](https://github.com/getcontacts/getcontacts). It has the benefit of being a very accurate method in that different cutoff distances and angles are used to identify contacts depending on the chemical groups of the residues involved in the contact.  

Both the HREMD simulation setup and contact calculations are being added to ChACRA via [femto](https://github.com/Psivant/femto) and [ProLif](https://prolif.readthedocs.io/en/stable/) as alternatives.

Assuming you have a 32 replica ensemble.

```

for i in {00..31}
do

                                 get_dynamic_contacts.py \
                                 --topology protein_system.pdb \
                                 --trajectory trajectories/rep_$i.xtc \
                                 --output contacts/rep_$i.tsv \
                                 --cores 36 \
                                 --itypes all --distout
done

for i in {00..31}
do
    get_contact_frequencies.py   --input_files contacts/rep_$i.tsv --output_file freqs/freqs_rep_$i.tsv
done

``` 

Produce dataframe wrapped by ContactFrequencies 

```
from ChACRA.chacra.ContactFrequencies import *

# point to the directory containing the contact frequency files
file_dir = 'path/to/files'
cont = ContactFrequencies(file_dir)

# you can save the dataframe and generate the cont object alternatively
cont.freqs.to_pickle('contact_frequencies.pd')
df = pd.read_pickle('contact_frequencies.pd')
cont = ContactFrequencies(df)

```

If you're dealing with a homomultimeric protein, you can average the contact frequencies to obtain more robust statistics.

```
from ChACRA.chacra.average import average_multimer
# return a dataframe of the averaged data
avg = average_multimer('structure.pdb',denominator=6,df=cont.freqs,representative_chains=['A','G'])

```

Then perform principal component analysis (PCA) to obtain the protein's "chacras". 

```

cpca = ContactPCA(avg)

```

The significant PCs can be identified with the difference of roots test[4].

```

from ChACRA.ContactAnalysis.plot import plot_difference_of_roots
# perform PCA on 500 scrambled contact data sets
cpca.permutated_explained_variance(avg, 500)
# the PCs that fall below the .05 p-value can be considered your baseline set of chacras pending further investigation  
plot_difference_of_roots(cpca)

```

Both the ContactPCA and the difference of roots test are performed automatically by default when creating the ContactFrequencies object and available as attributes of that class.

Project the data onto the principal components to visualize the energy-dependent trends of the chacras.
The lower eigenvalue modes (e.g. PCs 4 and above) can exhibit a decaying oscillatory pattern.  This is an artifact of the PCA; however, these modes' lowest temperature peak should coincide with the peaks seen in their highest loading score contacts. 

```

from ChACRA.ContactAnalysis.plot import plot_chacras
# 32 temperatures between 290 and 440 k
plot_chacras(cpca, temps=[i for i in np.geomspace(290,440,32)])

```
![chacras](https://github.com/Dan-Burns/ChACRA/assets/58605062/bfb0e0d1-6303-4683-a9a8-eb9daafac58f)


Now you can explore these PCs/chacras.

Contacts with relatively large absolute loading score values (the PC components) are highly energy-sensitive (within a given chacra).
These can be easily identified:

```

# find the sensitive contacts in PC1 (first chacra)
pc=1
cpca.sorted_norm_loadings(pc)

```


The resulting dataframe will have the indices (contacts) sorted in descending order of the absolute normalized value of the loading scores on the first PC.

Chacras can be visualized in pymol using .pml files generated with contacts_to_pymol.to_pymol.

![IGPS_chacras](https://github.com/Dan-Burns/ChACRA/assets/58605062/058f21d6-70a3-4360-b779-f5fff0066f18)


1. Burns, D., Singh, A., Venditti, V. & Potoyan, D. A. Temperature-sensitive contacts in disordered loops tune enzyme I activity. Proc. Natl. Acad. Sci. U. S. A. 119, e2210537119 (2022)

2. Burns, D., Venditti, V. & Potoyan, D. A. Temperature sensitive contact modes allosterically gate TRPV3. PLoS Comput. Biol. 19, e1011545 (2023)

3. Wang, L., Friesner, R. A. & Berne, B. J. Replica Exchange with Solute Scaling: A More Efficient Version of Replica Exchange with Solute Tempering (REST2). The Journal of Physical Chemistry B vol. 115 9431–9438 Preprint at https://doi.org/10.1021/jp204407d (2011)

4. Vieira, V. M. N. C. Permutation tests to estimate significances on Principal Components Analysis. Computational Ecology and Software 2, 103–123 (2012)

