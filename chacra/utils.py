from itertools import combinations
import pandas as pd
import numpy as np
import re
from MDAnalysis.analysis.distances import distance_array


def sort_dictionary_values(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: -item[1]))

def parse_id(contact):
    '''
    take the contact name (column id) and return a dictionary of
    the residue A descriptors and residue B descriptors
    '''
    chaina, resna, resida, chainb, resnb, residb = re.split(":|-", contact)
    return {'chaina':chaina, 'resna':resna, 'resida':resida,
             'chainb':chainb, 'resnb':resnb, 'residb':residb}

def split_id(contact):
    '''
    take the contact name and split it into its two residue parts
    returns a dictionary where 'resa' will contain 'CH:RES:NUM'
    '''
    resa, resb = re.split("-", contact)
    return {'resa':resa, 'resb':resb}




def get_angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle
        

