import re
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from scipy.stats import linregress
from ContactAnalysis.contact_functions import _parse_id, check_distance
import matplotlib as mpl
from pylab import cm
from matplotlib.colors import to_hex


# Refactoring of contacts_to_pymol_v2 (yes, v2 precedes this)

'''
There is probably a better way to do this than write a file with hundreds of selection strings.
Even if you still have to write the file, probably more organized to do things differently.
Could save a dictionary of dictionarys in the format of 
{'res1-res2': {'slope':-.5,
            {'color':'red',
            'line_width': 1,
            'dashes': True,
            'sphere_scale': .6,
            'sphere_transparency': 0,
            'sel_1': f'resname {resna} and resnum {resia} and chain {chaina}',
            'sel_2': f'resname {resnb} and resnum {resib} and chain {chainb}',
            'loading_score': .90,
            'top_PC': 1,

    }


You want to be able to hand it a list of contacts that maybe you put together by taking the top 10 of each PC and then 
all the prep happens behind the scenes.  

Only thing you tell it is if you want all the depictions limited to a single PC, or you want it to color them according 
to a list of PCs. 

Then you hand it either the contact frequency df or the contact obj and it puts it all together and organizes the 
contacts so that the higher scoring contacts are colored last.  

This can be done by sorting the dictionary according to the contact's loading scores in ascending order so the last lines written
are for contacts with loading score of 1.

Can make the slope depiction be more informative with dash gaps and line widths.


'''

def _get_slope(df,contact,temp_range=(0,7)):
    '''
    Return the slope of the contact frequency within the specified temperature
    (index) range.

    df: pd.DataFrame of contact frequencies 

    contact: str
        The name of the contact to retrieve the frequency slope for.
    
    temp_range: 2 int tuple
        The dataframe's index range to retrieve the frequency slope within.
    
    '''

    return linregress(df[contact].iloc[temp_range[0]:temp_range[1]].index, 
                   df[contact].iloc[temp_range[0]:temp_range[1]]).slope


def color_selection_by_pc(selection, contact_tuple):
    '''
    contact tuple can be generated from ContactPCA.get_top_contact()
    
    Need to add an argument to include a cmap and specify gradients
        something like np.linspace(min-max_freqs), cmap=plasma)
        and have it replace the values below.
    '''
    
    selection = re.split(' |,', selection)[1]
    
    if int(contact_tuple[-1]) == 1:
        color = 'red'
    elif int(contact_tuple[-1]) == 2:
        color = '0x02a8f8'
    elif int(contact_tuple[-1]) == 3:
        color = 'ytterbium'
    elif int(contact_tuple[-1]) == 4:
        color = 'purpleblue'
    elif int(contact_tuple[-1]) == 5:
        color = 'orange'
    elif int(contact_tuple[-1]) == 6:
        color = 'magenta'
    elif int(contact_tuple[-1]) == 7:
        color = 'gold'
   
    else:
        color = 'black'
    
    
    color_string = 'color ' + color + ', ' + selection
    
    return color_string