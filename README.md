# TSenCA

**T**emperature **Sen**sitive **C**ontact **A**nalysis

Tools to analyze biomolecular interactions across multiple ensembles.

After simulating a protein across multiple temperatures (ideally with replica exchange) use GetContacts to calculate the contact frequencies for all of the pairwise residue interactions.

Turn this data into a dataframe wrapped by the ContactFrequencies class to provide convenient methods to analyze the data.

Perform PCA on this data with the ContactPCA class and identify the temperature-dependent contact modes in your protein.

The PCs represent collective modes that describe different responses to temperature exhibited by different parts of the protein.  The most sensitive interactions (revealed by large magnitude loading scores) within the high eigenvalue PCs are strong candidates for functionally important residues.

The loading scores within a PC also report on correlated interactions and as a consequence, can reveal allostery.

You can make nice visualizations of the different PC mode interactions with contacts_to_pymol.to_visualize to help you grasp the complex dynamics of your protein.

See Burns, D., Singh, A., Venditti, V. & Potoyan, D. A. Temperature-sensitive contacts in disordered loops tune enzyme I activity. Proc. Natl. Acad. Sci. U. S. A. 119, e2210537119 (2022)
