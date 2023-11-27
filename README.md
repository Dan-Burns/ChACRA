# ChACRA![chacra_logo](https://github.com/Dan-Burns/ChACRA/assets/58605062/a030ffbb-0a97-4b33-a968-fab2ec7dbee9)




**Ch**emically **A**ccurate **C**ontact **R**esponse **A**nalysis

Find your protein's energy centers.

These are tools to analyze residue interactions across multiple ensembles.

After simulating a protein across multiple temperatures with replica exchange ("opening the chacras" ;D), use [GetContacts](https://getcontacts.github.io/interactions.html) to calculate the contact frequencies for all of the pairwise residue interactions.

Turn this data into a dataframe wrapped by the ContactFrequencies class to provide convenient methods to analyze the data.

Perform PCA on this data with the ContactPCA class and identify the temperature-dependent contact modes in your protein.

The PCs are collective modes (the "chacras") describing different responses to energy that characterize parts of the protein.  The most sensitive interactions (revealed by large magnitude loading scores) within the high eigenvalue PCs are strong candidates for functionally important residues.

The loading scores within a PC also report on correlated interactions and as a consequence, can reveal allostery.

You can make nice visualizations of the different PC mode interactions with contacts_to_pymol.to_visualize to help you grasp the complex dynamics of your protein.

If you use ChACRA, please cite:

[Burns, D., Singh, A., Venditti, V. & Potoyan, D. A. Temperature-sensitive contacts in disordered loops tune enzyme I activity. Proc. Natl. Acad. Sci. U. S. A. 119, e2210537119 (2022)](https://www.pnas.org/doi/10.1073/pnas.2210537119)

[Burns, D., Venditti, V. & Potoyan, D. A. Temperature sensitive contact modes allosterically gate TRPV3. PLoS Comput. Biol. 19, e1011545 (2023)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011545)
