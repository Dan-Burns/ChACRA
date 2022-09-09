# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import re
import pathlib
from sklearn.decomposition import PCA
from ContactAnalysis.contact_functions import _parse_id, check_distance, check_distance_mda
from scipy.stats import linregress
import MDAnalysis as mda



def make_contact_frequency_dictionary(freq_files):
    '''
    go through a list of frequency files and record all of the frequencies for 
    each replica.  
    '''
    
    
    contact_dictionary = {}
  
    regex = r'\w:\w+:\d+\s+\w:\w+:\d+'
    # go through each of the contact files and fill in the lists
    for i, file in enumerate(freq_files):
        with open(file, 'r') as freqs:
            for line in freqs.readlines():
                if re.search(regex, line):
                    line = line.strip()
                    first, second, num_str = line.split()
                    label = first + "-" + second
                    
                    
                    if label not in contact_dictionary.keys():
                        contact_dictionary[label] = [0 for n in range(i)]
                        contact_dictionary[label].append(float(num_str))
                    else:
                        contact_dictionary[label].append(float(num_str))
        
        #Extend all the lists before opening the next freq_file
        for key in contact_dictionary.keys():
            if i > 0 and len(contact_dictionary[key]) != i+1:
                length = len(contact_dictionary[key])
                extend = (i+1) - length
                contact_dictionary[key].extend([0 for n in range(extend)])
                
                    
    return contact_dictionary

class ContactFrequencies:
    
    
    
    def __init__(self, contact_data, temps=None):
        '''
        supply list of temperatures to replace index
        '''
        try:
            file_extension = pathlib.Path(contact_data).suffix
            if file_extension == '.csv':
                self.freqs = pd.read_csv(contact_data, index_col=0 )
            else:
                try:
                    self.freqs = pd.read_pickle(contact_data)
                except:
                    print('You can only use .csv or pickle format')
        except:
            self.freqs = contact_data
        
            
        if temps:
            mapper = {key:0 for key in self.freqs.index}
            for i,temp in enumerate(temps):
                mapper[i]=temp
            self.freqs = self.freqs.rename(mapper, axis=0)
    
    if __name__ == "__main__":
       pass
    
    def _parse_id(self, contact):
        '''
        take the contact name (column id) and return a dictionary of
        the residue A identfiers and residue B identifiers
        '''
        chaina, resna, resida, chainb, resnb, residb = re.split(":|-|\s+", contact)
        return {'chaina':chaina, 'resna':resna, 'resida':resida,
                 'chainb':chainb, 'resnb':resnb, 'residb':residb}
    
    def _split_id(self, contact):
        '''
        take the contact name and split it into its two residue parts
        '''
        resa, resb = re.split("-", contact)
        return {'resa':resa, 'resb':resb}
    
    def _get_slope(self,contact,temp_range=(0,7)):
        #TODO for networkx should combine slope and some min or max freq (b)
        return linregress(self.freqs[contact].iloc[temp_range[0]:temp_range[1]].index, 
                       self.freqs[contact].iloc[temp_range[0]:temp_range[1]]).slope
    
    
    def contact_partners(self, resid, resid_2=None, id_only=False):
        '''
        Provide a residue id and return the ids of all the residues it 
        makes contacts with
        '''
        # this should be able to deal with averaged data and original data 
        # and return any combination of residue+/chain+/resname
        
        contact_ids = []
        contact_names = []
        
        if id_only == True:
            for contact in self.freqs.columns:
                contact_info = self._parse_id(contact)
                if contact_info['resida'] == str(resid):
                    contact_ids.append(contact_info['residb'])
                elif contact_info['residb'] == str(resid):
                    contact_ids.append(contact_info['resida'])
        
            return contact_ids
        
        elif resid_2:
            for contact in self.freqs.columns:
                contact_info = self._parse_id(contact)
                if contact_info['resida'] == str(resid) and\
                   contact_info['residb'] == str(resid_2):
                    contact_names.append(contact)
                elif contact_info['residb'] == str(resid) and\
                   contact_info['resida'] == str(resid_2):
                    contact_names.append(contact)
            return contact_names
            
        else:
            for contact in self.freqs.columns:
                contact_info = self._parse_id(contact)
                if contact_info['resida'] == str(resid):
                    contact_names.append(contact)
                elif contact_info['residb'] == str(resid):
                    contact_names.append(contact)
            return contact_names
        
            
    
    def all_edges(self, weights=True, inverse=True, temp=0, as_dict=False):
        '''
        returns list of contact id tuples for network analysis input
        inverse inverts the edge weight so something with a high contact
        frequency has a low edge weight and is treated as if it is 
        'closer' in network analysis.
        '''
        all_contacts = []
        for contact in self.freqs.columns:
            partners = self._split_id(contact)
            if weights == True:
                if inverse == True:
                    weight = float(1/self.freqs[contact].loc[temp])
                else:
                    weight = float(self.freqs[contact].loc[temp])
                all_contacts.append((partners['resa'],
                                     partners['resb'], weight))
            
            else:
                all_contacts.append([partners['resa'], partners['resb']])
            
        if as_dict == True:
            if weights != True:
                print('Cannot use dictionary format without weights = True')
            contact_dict = {}
            for contact in all_contacts:
                contact_dict[(contact[0],contact[1])] = contact[2]
            return contact_dict
        
        else:
            return all_contacts

    def all_residues(self):
        all_residues = []
        for contact in self.freqs.columns:
            partners = self._split_id(contact)
            all_residues.append(partners['resa'])
            all_residues.append(partners['resb'])
        
        return all_residues
    
    def exclude_neighbors(self, n_neighbors=1):
        '''
        Reduce the contact dataframe to contacts separated by at least
        n_neighbors
        '''
        reduced_contacts = []
        for contact in self.freqs.columns:
            id_dict = self._parse_id(contact)
            # check this 
            if id_dict['chaina'] != id_dict['chainb']:
                continue
            else:
                if np.sqrt((int(id_dict['resida'])
                            -int(id_dict['residb']))**2) > n_neighbors:
                    reduced_contacts.append(contact)
        return reduced_contacts
    
    def average_contacts2(self, structure=None):
        '''
        Seems to work but doesn't seem any faster than original implementation.
        get contacts has the contact name arraged 'lexographically' so don't
        need to search for the equivalent contact with swapped name.
        Could use df.filter to get other column names
        '''
        # make a copy of the df 
        df = self.freqs.copy()
        
        if structure:
            u = mda.Universe(structure)
        
        averaged_data = {}
        # this will start a loop where after a column has been averaged,
        # the columns involved in the averaging will be dropped in place.
        while len(df.columns) > 0:
        
            resids = self._parse_id(df.columns[0])
            
            # If the contact is happening in the same subunit
            if resids['chaina'] ==  resids['chainb']:
                
                # Find others with matching residue numbers happening in the
                # same subunit
                to_average = [
                 c for c in df if 
                 (
                  ((re.split(':|-',c)[2] == resids['resida']) 
                  and 
                  (re.split(':|-',c)[5] == resids['residb']))
                  and
                  (re.split(':|-',c)[0] == re.split(':|-',c)[3])
                  )
                ]
                contact = 'A:'+resids['resna']+':'+resids['resida']+'-'+\
                'A:'+resids['resnb']+':'+resids['residb']
                
                averaged_data[contact] = df[to_average].mean(axis=1)
                # Get rid of the columns used for averaging
                df.drop(to_average, axis=1, inplace=True)
            # If they are happening inter-subunit
            # This does will not label opposing subunit contacts right
            # They'll be labeled as if they're happening on adjacent subunits
            # And if contacts are being made in the middle of the pore with
            # identical resids adjacent and opposite, they'll get combined for 
            # the time being. (not really interested in these rare contacts
            # that do not inform on the larger allosteric pathways atm)
            else:
                
                to_average = [
                 c for c in df if 
                 (
                  ((re.split(':|-',c)[2] == resids['resida']) 
                  and 
                  (re.split(':|-',c)[5] == resids['residb']))
                  and
                  (re.split(':|-',c)[0] != re.split(':|-',c)[3])
                  )
                ]
                
                # Give the general name to the inter-subunit contact
                contact = 'A:'+resids['resna']+':'+resids['resida']+'-'+\
                'B:'+resids['resnb']+':'+resids['residb']
                
                if structure:
                    
                    # find the closest residues in contact in the structure
                    contact = check_distance_mda(contact,u)
                    averaged_data[contact] = df[to_average].mean(axis=1)
                else:
                    averaged_data[contact] = df[to_average].mean(axis=1)
                    
                df.drop(to_average, axis=1, inplace=True)
            
        return pd.DataFrame(averaged_data)
                
            
            
    
    def average_contacts(self, structure=None):
        '''
        Very slow method.
        
        Might be improved by taking column 1 (maybe the way equivalent contacts
        (A/B or C/D) occur is always in same order since getcontacts formats it)
        and us regex to collect all columns with different chain ids, average
        them, then drop them in place from the df so it gets progressively smaller.
        
        Use this if all of your data comes from a protein with multiple
        identical subunits.  This will average the contacts that are identical/
        duplicated on each subunit.  Contacts that are only recorded 
        on one subunit will be included in the averaged contact df but will 
        be the same as their original values.
        
        Provide a structure to find intersubunit residues that actually 
        make contact so that they can be depicted properly (chain B res 1 with
        chain A res 50 vs chain B res 50 with chain A res 1..)
        '''
        # Make list of columns
        # For each column name, find any others that match the contacting indices and add as a list value to a dictionary
        # The lengths of all the dictionarys lists added together will equal the number of columns in the df if correct
    
        # then use the list of tuples to make a new dataframe where you average those columns
    
        # this hold the tuples showing column names with duplicated contacts
        duplicates = {}
        columns = list(self.freqs.columns)
    
        # This holds anything that gets appended to the hold list so the loop can skip them if they've already
        # been added to the duplicates dictionary
        check = []
    
        # iterate through the list of columns
        for i, column in enumerate(columns):
            # hold all the duplicated column ids for the current iteration
            hold = []
            resids1 = self._parse_id(column)
            # check to see if this column has already been included in a duplicate group
            if column in check:
                continue
            # if not continue the iteration looking for duplicates
            hold.append(column)
            check.append(column)
    
            # compare all of the subsequent columns remaining in the list to the current column
            for comparison_column in columns[i+1:]:
                resids2 = self._parse_id(comparison_column)
                # make sure that the comparison column has the same pattern of chains (same or dif)
                if (resids1['chaina'] == resids1['chainb'] and resids2['chaina'] == resids2['chainb']) or (
                    resids1['chaina'] != resids1['chainb'] and resids2['chaina'] != resids2['chainb']):
                    # if the chain pattern matches, the resids can match any order (could maybe use in new_list check)
                    if (resids1['resida'] == resids2['resida'] and resids1['residb'] == resids2['residb']) or (
                        resids1['resida'] == resids2['residb'] and resids1['residb'] == resids2['resida']):
                        hold.append(comparison_column)
                        check.append(comparison_column)
            # create a generalized key (for duplicates dictionary)
            # so that the duplicated columns and key name match the chain pattern
            # and make a list dictionary item with the column names
            # add a biopython pdb function to find shortest distance and name accordingly 
            # (if B to A < A to B then switch resids accordingly)
            if resids1['chaina'] == resids1['chainb']:
                duplicates['A:'+resids1['resna']+':'+resids1['resida']+'-'+
                          'A:'+resids1['resnb']+':'+resids1['residb']] = hold
            else:
                if structure:
                    contact = 'A:'+resids1['resna']+':'+resids1['resida']+'-'+\
                    'B:'+resids1['resnb']+':'+resids1['residb']
                    # find the closest residues in contact in the structure
                    duplicates[check_distance(contact,structure)] = hold
                else:
                    duplicates['A:'+resids1['resna']+':'+resids1['resida']+'-'+
                          'B:'+resids1['resnb']+':'+resids1['residb']] = hold
        #confirm that the dictionary contains the same number of column ids in the lists as the original dfs number of columns      
        counter = 0
        for item in duplicates.values():
            counter += len(item)
        #if counter == len(columns):
        #print('Number of columns in the duplicate column record matches the number of columns in the dataframe')
        
        # Now average the duplicate contact columns
        averaged_contacts = {key:0 for key in duplicates.keys()}
        for contact in duplicates.keys():
            averaged_contacts[contact]=self.freqs[duplicates[contact]].mean(axis=1)
        
        return pd.DataFrame(averaged_contacts) 

    def renumber_residues(self, starting_residue_number):   
        '''renumber the residues so the first residue begins with
        starting_residue_number.  Useful if the contact_files generated with
        get contacts was made with a incorrectly numbered structure file starting
        from 1.
        '''
        mapper = {}
        for column in self.freqs.columns:
            split_ids = self._parse_id(column)
            mapper[column] = split_ids['chaina']+':'+ split_ids['resna']+':'+\
                str(int(split_ids['resida'])+starting_residue_number-1)+'-'+\
                            split_ids['chainb']+':'+ split_ids['resnb']+':'+ \
                        str(int(split_ids['residb'])+starting_residue_number-1)
                        


    def exclude_below(self,min_frequency=0.05,temp_range=None):
        '''
        If the maximum frequency for a contact is below min_frequency,
        remove it from the dataset.
        '''
        if temp_range:
            return self.freqs[(self.freqs.iloc[temp_range[0]:temp_range[1]].max() 
                              > min_frequency).index[self.freqs.iloc[
                              temp_range[0]:temp_range[1]].max() > 
                               min_frequency]]
        else:
            return self.freqs[(self.freqs.max() > min_frequency).index[
                    self.freqs.max() > min_frequency]]
        
    def exclude_above(self,max_frequency=0.98):
        '''
        If the minimum frequency for a contact is above max_frequency,
        remove it from the dataset.
        '''
        return self.freqs[(self.freqs.min() < max_frequency).index[
                self.freqs.min() < max_frequency]]

        
    def shortest_route(self, structure, begin_res, end_res):
        '''
        Use the contact labels and the structure to find the shortest
        route between two residues. 
        Backbone should probably be calculated.
        Consider the strengths of the contacts (frequencies) as well to find
        the strongest route.
        This does not guarantee that the route will be continguous since
        one residue might have to exchange contacts between two others.
        '''
                                                        
                                       


def _normalize(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''
    Normalize the loading score dataframe
    '''
    result = df.copy()
    for pc in df.columns:
        result[pc] = df[pc].abs()/df[pc].abs().max()
    return result       
    
class ContactPCA:
    '''
    Class takes a ContactFrequency object and performs principal component 
    analysis on it. 
    '''
    def __init__(self, contact_df, n_components=.999):
        pca = PCA(n_components=n_components)
        self.pca = pca.fit(contact_df)
        self.loadings = pd.DataFrame(self.pca.components_.T, columns=
                        ['PC'+str(i+1) for i in range(np.shape
                         (pca.explained_variance_ratio_)[0])], 
                        index=list(contact_df.columns))
        self.norm_loadings = _normalize(self.loadings)
        
    def _split_id(self, contact):
        '''
        take the contact name and split it into its two residue parts
        '''
        resa, resb = re.split("-", contact)
        return {'resa':resa, 'resb':resb}

    def sorted_loadings(self, pc=1):
       
        return self.loadings.iloc[(-self.loadings['PC'+str(pc)].abs())
                                  .argsort()]
        
    def sorted_norm_loadings(self, pc=1):    
        return self.norm_loadings.iloc[(-self.norm_loadings['PC'+str(pc)]
                                                            .abs()).argsort()]
    
    def edges(self, weights=True, pc=1, percentile=99):
        '''
        edit for PCA df format
        '''
        percentile_df = self.sorted_loadings(pc).loc[
                        self.sorted_loadings(pc)['PC'+str(pc)] >
                        np.percentile(self.sorted_loadings(pc)[
                                'PC'+str(pc)],percentile)]
        edges = []
        for contact in percentile_df.index:
            partners = self._split_id(contact)
            if weights == True:
                weight = float(percentile_df['PC'+str(pc)].loc[contact])
                edges.append((partners['resa'],
                                     partners['resb'], weight))
            else:
                edges.append((partners['resa'], partners['resb']))
        
        return edges   

    
    def all_edges(self, weights=True, pc=1):
        '''
        edit for PCA df format
        '''
        all_contacts = []
        for contact in self.loadings.index:
            partners = self._split_id(contact)
            if weights == True:
                weight = float(self.loadings['PC'+str(pc)].loc[contact])
                all_contacts.append((partners['resa'],
                                     partners['resb'], weight))
            else:
                all_contacts.append((partners['resa'], partners['resb']))
        
        return all_contacts   

                
    def get_top_contact(self, resnum, pc_range=(1,5)):
        '''
        Return the contact name, normalized loading score, pc on which it has 
        its highest score, and the overall rank the score represents on the pc.
        pc_range is the range of PCs to include in the search for the
        highest score
        '''
        pc_range = range(pc_range[0],pc_range[1])
        pcs = ['PC'+str(i) for i in pc_range]
        
        # list to store the contact ids involving resnum
        contacts = []
        # find the contact ids involving resnum
        for contact in self.norm_loadings.index:
            if str(resnum) in _parse_id(contact).values():
                contacts.append(contact)
        if contacts == []:
            return None
        # return the highest scoring contact loading score among pc_range
        # involving resnum
        highest_score = self.norm_loadings[pcs].loc[contacts].max().max()
        
        # get the PC that the highest score occurs
        for label in self.norm_loadings[pcs].loc[contacts].max().index:
            if self.norm_loadings[pcs].loc[contacts].max().loc[label] == highest_score:
                highest_pc = label[2:]
                # get the contact id having the highest score
                for contact in contacts:
                    if self.norm_loadings[label].loc[[contact]][0] == highest_score:
                        highest_scoring_contact = contact
                        
        rank = self.sorted_norm_loadings(highest_pc)[['PC'+highest_pc]
                        ].index.to_list().index(highest_scoring_contact)+1
        
        return (highest_scoring_contact, highest_score, highest_pc, str(rank))
    
    def get_scores(self, contact, pc_range=range(1,5)):
        '''
        Return the normalized loading score, pc,
        rank on each pc in pc_range
        '''
        contacts = []
        for pc in pc_range:
            rank = list(self.sorted_norm_loadings(pc).index
                                   ).index(contact) +1
            score = (self.sorted_norm_loadings(pc)['PC'+str(pc)].loc[contact])
            
            percentile = 100 - ((rank/len(self.loadings))*100)
            contacts.append({'score':score, 'PC':pc, 'rank':rank,
                             'percentile':percentile})
            
         
        return contacts
            
        
    def in_percentile(self, contact, percentile, pc=None):
        '''Provide a contact and a percentile cutoff to consider the top range
        and the pc to search and return True if the contact falls in the top 
        range on that pc.
        '''
        
        percentile_df = self.sorted_norm_loadings(pc).loc[
                        self.sorted_norm_loadings(pc)['PC'+str(pc)] >
                        np.percentile(self.sorted_norm_loadings(pc)[
                                'PC'+str(pc)],percentile)]
                
        
        if contact in percentile_df['PC'+str(pc)].index:
            return True
        else:
            return False
    
    
            
                





