

# break up average contacts into manageable functions
from itertools import combinations

def find_identical_subunits(universe):
    '''
    determines which subunits are identical
    '''
    residues = {seg.segid: seg.residues.resnames for seg in universe.segments}
    segids = list(residues.keys())
    array = np.zeros((len(segids),len(segids)),dtype=bool)
    np.fill_diagonal(array,True)
    identical_table = pd.DataFrame(array, columns=segids, index=segids)

    for combo in combinations([segid for segid in segids],2):
        bool = np.all(np.equal(residues[combo[0]],residues[combo[1]]))
        identical_table[combo[1]][combo[0]], identical_table[combo[0]][combo[1]] =  bool, bool
    
    identical_table.drop_duplicates(inplace=True)

    identical_subunits = {}
    for i, segid in enumerate(identical_table.index):
        subunits = identical_table.T[identical_table.loc[segid]==True].index
        if len(subunits) >= 2:
            identical_subunits[i] = list(subunits)

    return identical_subunits