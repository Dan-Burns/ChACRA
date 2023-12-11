# -*- coding: utf-8 -*-
"""
Author: Dan Burns
"""
import pandas as pd
import numpy as np
import re
import os
import pathlib
from sklearn.decomposition import PCA
from .utils import *
import MDAnalysis as mda
import collections
from .utils import *
import tqdm




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
    
    
    
    def __init__(self, contact_data, temps=None, temp_progression=None, min_max_temp=None):
        '''
        TODO supply endpoints or list of temperatures and replace index
        TODO supply path to freq files and make everything.

        contact_data : string or pd.DataFrame or dict
            Path to file ('.csv') or pickle ('.pd') of the prepared contact frequency data or the path to the directory
            containing the getcontacts '.tsv' frequency files or a dictionary or dataframe containing the contact frequencies.

        temps : list
            Option to specify all of the index values as temperature provided in temps.

        temp_progression : string
            'linear' or 'geometric' 
            If no list of temps is provided, make a list of temperatures between min_max_temp values in either
            linear or geometric progression.
        
        min_max_temp : tuple of int or float
            The highest and lowest temperatures to interpolate between with temp_progression.

        Returns
        -------
        A ContactFrequencies object that wraps a pd.DataFrame with conventient methods to investigate contact frequencies.
    
        '''
        try:
            if os.path.isfile(contact_data):
            
                
                file_extension = pathlib.Path(contact_data).suffix
                if file_extension == '.csv':
                    self.freqs = pd.read_csv(contact_data, index_col=0 )

                elif file_extension == '.pd':
                    self.freqs = pd.read_pickle(contact_data)
                else:
        
                    print('Provide a file with a .csv or .pd (pickle format) file extension. \n'\
                            'Or provide a path the folder containing the original getcontacts .tsv files.')
            
            elif os.path.isdir(contact_data):

                contact_files = [f'{contact_data}/{file}' for file in sorted(os.listdir(contact_data)) if file.endswith('.tsv')]
                print("Arranging the data in the following order :", flush=True)
                for file in contact_files:
                    print(file, flush=True)
                contact_dictionary = make_contact_frequency_dictionary(contact_files)
                self.freqs = pd.DataFrame(contact_dictionary)
        except:
            try:
                # assuming that you're handing it a dataframe or dictionary
                if type(contact_data) == pd.DataFrame:
                    self.freqs = contact_data
                elif type(contact_data) == dict:
                    self.freqs = pd.DataFrame(contact_data)
            except:
                ("Provide one of file (.pd or .csv extension), pd.DataFrame, dict, or path to original getcontacts .tsv frequency files.")
            
        
            
        if temps:
            mapper = {key:temp for key,temp in zip(self.freqs.index, temps)}
            
            self.freqs = self.freqs.rename(mapper, axis=0)
        elif temp_progression is not None and min_max_temp is not None:
            if temp_progression == 'linear':
                temps = [int(i) for i in np.linspace(min_max_temp[0], min_max_temp[1], len(self.freqs))]
            elif temp_progression == 'geometric':
                temps = [int(i) for i in geometric_progression(min_max_temp[0], min_max_temp[1], len(self.freqs))]
            mapper = {key:temp for key,temp in zip(self.freqs.index, temps)}
            
            self.freqs = self.freqs.rename(mapper, axis=0)
        

    
    if __name__ == "__main__":
       pass
    
    
    def get_contact_partners(self, resid1, resid2=None, ):
        '''
        Filter the dataframe to only return contacts involving the provided residue(s).

        Parameters
        ----------

        resid1 : int or tuple (chain, resname, resid)
            If providing a single integer, returns all contacts involving that resid.
            If tuple, must contain integer resid in third position.  Chain and 
            resname are optional.
        resid2 : int
            If resid2 is provided, resid1 can only take an integer value

        Returns
        -------
        DataFrame 
        DataFrame filtered to only contacts involving resid(s)
        
        #TODO allow for chain and id only or resn only
        '''

        if resid2 is not None:
            if type(resid2) == tuple:
                chain2 = resid2[0]
                resn2 = resid2[1]
                resid2 = resid2[2]
            else:
                chain2 = '[A-Z1-9]+'
                resn2 = '[A-Z]+'
            if type(resid1) == tuple:
                chain = resid1[0]
                resn = resid1[1]
                resid1 = resid1[2]
            else:
                chain = '[A-Z1-9]+'
                resn = '[A-Z]+'
            regex1 = f"{chain}:{resn}:{resid1}(?!\d)-{chain2}:{resn2}:{resid2}(?!\d)"
            regex2 = f"{chain2}:{resn2}:{resid2}(?!\d)-{chain}:{resn}:{resid1}(?!\d)"
            regex = f"{regex1}|{regex2}"
        else:
            if type(resid1) == tuple:
                chain = resid1[0]
                resn = resid1[1]
                resid1 = resid1[2]
            else:
                chain = '[A-Z1-9]+'
                resn = '[A-Z]+'
            regex1 = f"{chain}:{resn}:{resid1}(?!\d)-[A-Z1-9]+:[A-Z]+:\d+"
            regex2 = f"[A-Z1-9]+:[A-Z]+:\d+-{chain}:{resn}:{resid1}(?!\d)"
            regex = f"{regex1}|{regex2}"
        return self.freqs.filter(regex=regex, axis=1)
    
      
    def get_all_edges(self, weights=True, inverse=True, temp=0, index=None, as_dict=False):
        '''
        returns list of contact id tuples for network analysis input
        inverse inverts the edge weight so something with a high contact
        frequency has a low edge weight and is treated as if it is 
        'closer' in network analysis.

        Parameters
        ----------
        weights : bool
            If True, return the connected nodes as well as the associated edge weight.

        inverse : bool
            Whether or not to return the inverse edge weight (inverse contact frequency) or original contact
            frequency as the edge weight.

        temp : int
            The row (.loc) from which to collect the data.

        index : None or int
            The row from which to collect the data. If temp is provided, the index (.iloc) overrides the temp.

        as_dict : bool
            If True, return the data in dictionary formate with tuple node names as keys and edge weight values.
        '''
        if as_dict == True:
            weights = True
        all_contacts = []
        for contact in self.freqs.columns:
            partners = split_id(contact)
            if weights == True:
                if inverse == True:
                    if index is not None:
                        weight = float(1/self.freqs[contact].iloc[index])
                    else:
                        weight = float(1/self.freqs[contact].loc[temp])
                else:
                    if index is not None:
                        weight = float(self.freqs[contact].iloc[temp])
                    else:
                        weight = float(self.freqs[contact].loc[temp])
                all_contacts.append((partners['resa'],
                                     partners['resb'], weight))
            
            else:
                all_contacts.append([partners['resa'], partners['resb']])
            
        if as_dict == True:
            contact_dict = {}
            for contact in all_contacts:
                contact_dict[(contact[0],contact[1])] = contact[2]
            return contact_dict
        
        else:
            return all_contacts

    def get_all_residues(self):
        all_residues = []
        for contact in self.freqs.columns:
            partners = split_id(contact)
            all_residues.append(partners['resa'])
            all_residues.append(partners['resb'])
        
        return all_residues
    
    def exclude_neighbors(self, n_neighbors=1):
        '''
        Reduce the contact dataframe contacts list to those separated by at least
        n_neighbors
        '''
        reduced_contacts = []
        for contact in self.freqs.columns:
            id_dict = parse_id(contact)
            # check this 
            if id_dict['chaina'] != id_dict['chainb']:
                continue
            else:
                if np.abs(int(id_dict['resida'])
                            -int(id_dict['residb'])) > n_neighbors:
                    reduced_contacts.append(contact)
        return reduced_contacts


    def renumber_residues(self, starting_residue_number):   
        '''renumber the residues so the first residue begins with
        starting_residue_number.  Useful if the contact_files generated with
        get contacts was made with a incorrectly numbered structure file starting
        from 1.
        '''

        # TODO add option to renumber from several starting points/chains
        mapper = {}
        for column in self.freqs.columns:
            split_ids = parse_id(column)
            mapper[column] = split_ids['chaina']+':'+ split_ids['resna']+':'+\
                str(int(split_ids['resida'])+starting_residue_number-1)+'-'+\
                            split_ids['chainb']+':'+ split_ids['resnb']+':'+ \
                        str(int(split_ids['residb'])+starting_residue_number-1)
        # TODO actually return the renumbered dataframe
                


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


    def to_heatmap(self,format='mean', range=None, contact_pca=None, pc=None):

        '''
        Convert the deata into a symmetric matrix with residues mirrored on x and y axis.
        The residues will be named as "chainResid" e.g. A100.  You can easily plot the 
        heatmap with seaborn.heatmap().  If you want a heatmap of a subset of residues,
        filter the contact dataframe to your contacts of interest and generate a new
        ContactFrequencies object with the filtered dataframe before calling to_heatmap().

        Parameter
        '''
        
        # Turn the data into a heatmap 
        # format options are 'mean', 'stdev', 'difference', 'loading_score'
        # if 'difference', specify tuple of rows your interested in taking the difference from
        # if 'loading_score' then specify contact_pca data and pc from which you want the loading score
        # hold reslists with chain keys and list of resid values
        reslists = {}
        res_append = lambda res: f"{chain}{res}"
        for contact in self.freqs.columns:
            resinfo = parse_id(contact)

            if resinfo['chaina'] in reslists.keys():
                reslists[resinfo['chaina']].append(int(resinfo['resida']))
            else:
                reslists[resinfo['chaina']] = [int(resinfo['resida'])]
            if resinfo['chainb'] in reslists.keys():
                reslists[resinfo['chainb']].append(int(resinfo['residb']))
            else:
                reslists[resinfo['chainb']] = [int(resinfo['residb'])]
        
        # sort the dictionary by the chain id keys
        reslists = {key: reslists[key] for key in sorted(reslists.keys())}
        
        # eliminate duplicates, sort the reslists in ascending order, and make a single list of all resis
        all_resis = []
        for chain in reslists:
            reslists[chain] = list(set(reslists[chain]))
            reslists[chain].sort()
            # map the chain id onto the resid
            # this will be the indices and columns for the heatmap
            # lambda function for mapping chain id back onto residue
            all_resis.extend(list(map(res_append,reslists[chain])))

        # create an empty heatmap
        data = np.zeros((len(all_resis), len(all_resis)))

        # get the index for the corresponding residue
        for contact in self.freqs.columns:
            resinfo = parse_id(contact)
            index1 = all_resis.index(f"{resinfo['chaina']}{resinfo['resida']}")
            index2 = all_resis.index(f"{resinfo['chainb']}{resinfo['residb']}")

            values = {}
            values['mean'], values['stdev'], values['difference'] = self.freqs[contact].mean(), self.freqs[contact].std(), np.abs(self.freqs[contact].iloc[-1])-np.abs(self.freqs[contact].iloc[0])
            if contact_pca:
                #TODO offer sorted loadings to catch sign
                values['loading_score'] = contact_pca.sorted_norm_loadings(pc)[f'PC{pc}'].loc[contact]

            data[index1][index2] = values[format]
            data[index2][index1] = values[format]
        
        return pd.DataFrame(data, columns=all_resis, index=all_resis)           

def _de_correlate_df(df):
    '''
    randomize the values within a dataframe's columns
    '''
    
    a = df.values
    idx = np.random.rand(*a.shape).argsort(0) # argsort(0) returns row indices
    out = a[idx, np.arange(a.shape[1])] # index a by independently randomized rows and original column order
    return pd.DataFrame(out, columns=df.columns)                  

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

    #TODO ensure that signs of variables correspond to expected melting trend of PC1
    def __init__(self, contact_df):
        pca = PCA()
        print("Opening the chacras.")
        self.pca = pca.fit(contact_df)
        self._transform = pca.transform(contact_df)
        self.loadings = pd.DataFrame(self.pca.components_.T, columns=
                        ['PC'+str(i+1) for i in range(np.shape
                         (pca.explained_variance_ratio_)[0])], 
                        index=list(contact_df.columns))
        self.norm_loadings = _normalize(self.loadings)
        self._permutated_explained_variance = None

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
        #TODO this needs to be investigated - contact frequencies are the natural edge choice
        # perhaps loading scores will be useful as edges
        percentile_df = self.sorted_loadings(pc).loc[
                        self.sorted_loadings(pc)['PC'+str(pc)] >
                        np.percentile(self.sorted_loadings(pc)[
                                'PC'+str(pc)],percentile)]
        edges = []
        for contact in percentile_df.index:
            partners = split_id(contact)
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
            partners = split_id(contact)
            if weights == True:
                weight = float(self.loadings['PC'+str(pc)].loc[contact])
                all_contacts.append((partners['resa'],
                                     partners['resb'], weight))
            else:
                all_contacts.append((partners['resa'], partners['resb']))
        
        return all_contacts   

                
    def get_top_contact(self, resnum, pc_range=(1,4)):
        '''
        Return the contact name, normalized loading score, pc on which it has 
        its highest score, and the overall rank the score represents on the pc.
        pc_range is the range of PCs to include in the search for the
        highest score
        
        pc_range is inclusive
        '''
        pcs = ['PC'+str(i) for i in range(pc_range[0],pc_range[1]+1)]
        contacts = []
        for contact in self.norm_loadings.index:
            #TODO add resname and/or chain 
            if str(resnum) in parse_id(contact).values():
                contacts.append(contact)


        highest_scores = self.norm_loadings[pcs].loc[contacts].max()
        top_score = highest_scores.sort_values()[-1]
        top_pc = highest_scores.sort_values().index[-1]
        contact = self.norm_loadings.loc[contacts][self.norm_loadings[pcs].loc[contacts][top_pc] == top_score].index[0]
        return {'contact':contact, 'PC':top_pc, 'loading_score':top_score}
    
    
    ## TODO this is slow - minutes to run on the entire contact list
    def get_scores(self, contact, pc_range=(1,4)):
        '''
        Return the normalized loading score,
        rank, and percentile it falls in for the contact on each pc in pc_range
        dictionary keys are PC numbers corresponding to dictionaries of these
        items
        pc_range is inclusive
        '''

        pc_range = range(pc_range[0],pc_range[1]+1)

        contacts = {pc:{} for pc in pc_range}
        for pc in pc_range:
            
            contacts[pc]['rank'] = list(self.sorted_norm_loadings(pc).index
                                   ).index(contact) +1
            contacts[pc]['score'] = (self.sorted_norm_loadings(pc)['PC'+str(pc)].loc[contact])
            
      
        # sort the dictionary by score
        result = collections.OrderedDict(sorted(contacts.items(), key=lambda t:t[1]["score"]))
        # put in descending order
        return collections.OrderedDict(reversed(list(result.items())))
        
            
        
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
        
    def get_chacra_centers(self, pc, cutoff=0.6, absolute=True):
        '''
        Return the loading score dataframe containing only the contacts with loading scores above the cutoff

        Parameters
        ----------
        pc : int
            The principal component from which to retrieve the most responsive contacts.

        cutoff : float
            The minimum absolute value of loading score to include in the top sensitive contacts

        absolute : bool
            Whether to return the dataframe with normalized absolute values or original loading scores 
        '''

    
        chacra_centers = self.sorted_norm_loadings(pc).loc[(self.sorted_norm_loadings(pc)[f'PC{pc}'] >= cutoff)].index
        if absolute == True:
            return self.sorted_norm_loadings(pc).loc[chacra_centers]
        else:
            return self.loadings.loc[chacra_centers]
        
   

    def permutated_pca(self, contact_frequencies, N_permutations=200, get_loading_pvals=False):
        '''
        Randomize the values within the contact frequency columns to test the significance of the contact PCs.

        contact_frequencies : pd.DataFrame
            The dataframe of contact frequencies that the ContactPCA is based off of.

        N_permutations : int
            Number of times to randomize the dataframe and perform PCA.

        Returns
            np.array of explained variance by PC for each permutation of the dataframe.
        '''    
        # borrowed code from here https://www.kaggle.com/code/tiagotoledojr/a-primer-on-pca
        self._N_permutations = N_permutations
        df = contact_frequencies.copy()
        # This function changes the order of the columns independently to remove correlations
       
        #original_variance = self.pca.explained_variance_ratio_
        pca = PCA()
        variance = np.zeros((N_permutations, len(df.index)))
        print('This can take a moment.')
        for i in tqdm.tqdm(range(N_permutations)):
            X_aux = _de_correlate_df(df)    
            pca.fit(X_aux)
            # record the explained variance of this iteration's PCs
            variance[i, :] = pca.explained_variance_ratio_
            # track the behavior of the loading scores too
            if get_loading_pvals:
                if i == 0:
                    permutated_components = (np.abs(pca.components_) >= np.abs(self.pca.components_))*1
                else:
                    # summing up the number of times that the randomized data loadings have greater values than the 
                    # real data
                    permutated_components += (np.abs(pca.components_) >= np.abs(self.pca.components_))*1
        self._permutated_explained_variance = variance
        # The average of this tells you the probability of randomized data having higher loading scores
        # if this value is low (<0.05) then the real loading score is significant
        if get_loading_pvals:
            self.permutated_component_pvals = permutated_components/N_permutations
    

    #######################
    # def bootstrapped_pca(self):
    #     '''
        
    #     '''

    #     # Bootstrap
    #     # Empirical loadings
    #     loadings = cpca.pca.components_.T 
    #     nboot=1000
    #     # Bootstrap samples
    #     loadings_boot = []
    #     for i in range(nboot):
    #         X_boot = df.sample(df.shape[0], replace=True) 
    #         pca_boot = PCA().fit(X_boot)
    #         loadings_boot.append(np.abs(pca_boot.components_.T)>=np.abs(loadings))
    #     pvals = np.dstack(loadings_boot).mean(axis=2) 