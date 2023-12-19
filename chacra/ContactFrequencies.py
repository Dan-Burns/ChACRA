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
from .utils import *
import tqdm
from scipy.stats import linregress 




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
    
    
    
    def __init__(self, contact_data, 
                 temps=None, 
                 temp_progression=None, 
                 min_max_temp=None, 
                 structure=None,
                 get_chacras=True, 
                 N_permutations=500):
        '''
        The main object for investigating the molecule's contact frequency data.

        Parameters
        ----------

        contact_data : string or pd.DataFrame or dict
            Path to file ('.csv') or pickle ('.pd') of the prepared contact 
            frequency data or the path to the directory containing the 
            getcontacts '.tsv' frequency files or a dictionary or dataframe 
            containing the contact frequencies.

        temps : list
            A list specifying all of the dataframe's index values as temperatures.

        temp_progression : string
            Options are 'linear' or 'geometric'. 
            If no list of temps is provided, make a list of temperatures between
            min_max_temp values in either linear or geometric progression.
        
        min_max_temp : tuple of int or float
            The highest and lowest temperatures to interpolate between with 
            temp_progression to create the dataframe's index.
        
        structure : str
            Path to the structure file that the contact data is based on.
            Conventiently makes the structure available to other functions. 

        get_chacras : bool
            Make the ContactPCA class an attribute of this object.  
            i.e. cont = ContactFrequencies(data,get_chacras=True)
                cont.cpca.get_chacra_centers(1)...
        
        N_permutations : int
            If get_chacras == True, the number of times to permutate the data to
            obtain chacra (PC) significance values.

        Returns
        -------
        A ContactFrequencies object that wraps a pd.DataFrame with conventient 
        methods to investigate contact frequencies.
    
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

                contact_files = [f'{contact_data}/{file}' for file in sorted(
                            os.listdir(contact_data)) if file.endswith('.tsv')]
                print("Arranging the data in the following order :", flush=True)
                for file in contact_files:
                    print(file, flush=True)
                contact_dictionary = make_contact_frequency_dictionary(
                                                            contact_files)
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
                temps = [int(i) for i in np.linspace(min_max_temp[0], 
                                            min_max_temp[1], len(self.freqs))]
            elif temp_progression == 'geometric':
                temps = [int(i) for i in np.geomspace(min_max_temp[0], 
                                            min_max_temp[1], len(self.freqs))]
            mapper = {key:temp for key,temp in zip(self.freqs.index, temps)}
            
            self.freqs = self.freqs.rename(mapper, axis=0)
        
        if structure:
            self.structure = structure

        # give this object access to the ContactPCA 
        if get_chacras == True:
            self.cpca = ContactPCA(self.freqs, N_permutations=N_permutations)
        else:
            self.cpca = None
        

    
    def get_contact_partners(self, resid1, resid2=None, ):
        '''
        Filter the dataframe to only return contacts involving the 
        provided residue(s).

        Parameters
        ----------

        resid1 : int or tuple (chain, resname, resid)
            If providing a single integer, returns all contacts involving that 
            resid. If tuple, must contain integer resid in third position.  
            Chain and resname are optional.
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
    
      
    def get_edges(self, weights=True, inverse=True, temp=0, index=None, as_dict=False):
        '''
        returns list of contact id tuples for network analysis input
        inverse inverts the edge weight so something with a high contact
        frequency has a low edge weight and is treated as if it is 
        'closer' in network analysis.

        Parameters
        ----------
        weights : bool
            If True, return the connected nodes as well as the associated edge 
            weight.

        inverse : bool
            Whether or not to return the inverse edge weight (inverse contact 
            frequency) or original contact frequency as the edge weight.

        temp : int
            The row (.loc) from which to collect the data.

        index : None or int
            The row from which to collect the data. If temp is provided, the 
            index (.iloc) overrides the temp.

        as_dict : bool
            If True, return the data in dictionary formate with tuple node names
            as keys and edge weight values.
        
        Returns
        -------
        List of tuples or Dict of tuple node names and weight values
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
        '''
        Returns a list of all the residues
        Not used.
        '''
        return list(set(
            [a for cont in self.freqs.columns for a in split_id(cont).values()]))
   
    
    def exclude_neighbors(self, n_neighbors=1):
        '''
        Reduce the contact dataframe contacts to those separated by at least
        n_neighbors. Returns a list of contact ids.

        n_neighbors : int
            Number of residues that must separate contacting residues in the
            sequence.
        
        Returns
        -------
        list of contact (column) names.
        '''
        reduced_contacts = []
        for contact in self.freqs.columns:
            id_dict = parse_id(contact)
            # resid can be same or within n_neighbors if chain id is different
            if id_dict['chaina'] != id_dict['chainb']:
                continue
            else:
                if np.abs(int(id_dict['resida'])
                            -int(id_dict['residb'])) > n_neighbors:
                    reduced_contacts.append(contact)
        return reduced_contacts


    # def renumber_residues(self, starting_residue_number):   
    #     '''
    #     NOT IMPLEMENTED
    #     Doesn't account for different chains with different starting ids.

    #     renumber the residues so the first residue begins with
    #     starting_residue_number.  Useful if the contact_files generated with
    #     get contacts was made with a incorrectly numbered structure file starting
    #     from 1.
    #     '''

    #     # TODO add option to renumber from several starting points/chains
    #     mapper = {}
    #     for column in self.freqs.columns:
    #         split_ids = parse_id(column)
    #         mapper[column] = split_ids['chaina']+':'+ split_ids['resna']+':'+\
    #             str(int(split_ids['resida'])+starting_residue_number-1)+'-'+\
    #                         split_ids['chainb']+':'+ split_ids['resnb']+':'+ \
    #                     str(int(split_ids['residb'])+starting_residue_number-1)
    #     # TODO return the renumbered dataframe in place.
                


    def exclude_below(self,min_frequency=0.05,row_range=None):
        '''
        If the maximum frequency for a contact is below min_frequency,
        remove it from the dataset.

        Parameters
        ----------
        min_frequency : float
            Cutoff contact probability.  If a contact does not exceed this value
            at any temperature (row) then it is excluded from the returned 
            dataframe.
        
        row_range : tuple
            A 2 integer tuple corresponding to the first a last+1 rows to 
            consider when excluding contacts that do not exceed the cutoff.

        Returns
        -------
        pd.DataFrame of contacts meeting the cutoff criteria
        '''
        if row_range:
            return self.freqs[(self.freqs.iloc[row_range[0]:row_range[1]].max() 
                              > min_frequency).index[self.freqs.iloc[
                              row_range[0]:row_range[1]].max() > 
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


    def to_heatmap(self,format='mean', pc=None):

        '''
        Convert the data into a symmetric matrix with residues mirrored on x 
        and y axis. The residues will be named as "chainResid" e.g. A100.  
        You can plot the returned array with seaborn.heatmap().  

        If you want a heatmap of a subset of residues, filter the contact 
        dataframe to your contacts of interest and generate a new 
        ContactFrequencies object with the filtered dataframe before calling 
        to_heatmap().

        Parameters
        ----------
         
        format : str
            format options are 'mean', 'stdev', 'difference', 'loading_score'.
            If 'difference', specify tuple of rows your interested in taking the 
            difference from.
            If 'loading_score', then specify contact pc/chacra from which you 
            want the loading score. cpca attribute must be available.

        Returns
        -------
        pd.DataFrame
        '''
        
        #hold reslists with chain keys and list of resid values
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
        
        # eliminate duplicates, sort the reslists in ascending order, and make a
        # single list of all resis
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
            values['mean']  = self.freqs[contact].mean()  
            values['stdev'] = self.freqs[contact].std()
            values['difference'] = np.abs(self.freqs[contact].iloc[-1])-np.abs(self.freqs[contact].iloc[0])
            if hasattr(self, "cpca") and pc is not None:
                #TODO offer sorted loadings to catch sign
                values['loading_score'] = self.cpca.sorted_norm_loadings(pc)[f'PC{pc}'].loc[contact]
            elif pc is not None and hasattr(self, "cpca") == False:
                print("Instantiate the cpca attribute with ContactPCA.")

            data[index1][index2] = values[format]
            data[index2][index1] = values[format]
        
        return pd.DataFrame(data, columns=all_resis, index=all_resis)           

def _de_correlate_df(df):
    '''
    randomize the values within a dataframe's columns
    '''
    # improved version!
    a = df.values
    idx = np.random.rand(*a.shape).argsort(0) # argsort(0) returns row indices
    out = a[idx, np.arange(a.shape[1])] # index by independently randomized rows
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
    Performs PCA on the contact frequency data and provides methods to 
    investigate the chacras.

    Parameters
    ----------
    contact_df : pd.DataFrame
        The contact frequency dataframe.

    significance_test : bool
        Perform the difference of roots test to identify the significant
        principal components / chacras.

    N_permutations : int
        The number of times to randomize the data and perform PCA for the 
        signficance test.
    '''
    
    
    def __init__(self, contact_df, significance_test=True, N_permutations=500):
        #TODO allow for ContactFrequencies input
        pca = PCA()
        print("Opening the chacras.")
        self.pca = pca.fit(contact_df)
        self._transform = pca.transform(contact_df)
        self.loadings = pd.DataFrame(self.pca.components_.T, 
                        columns=['PC'+str(i+1) for i in range(np.shape
                         (pca.explained_variance_ratio_)[0])], 
                        index=list(contact_df.columns))
        self.norm_loadings = _normalize(self.loadings)

        # ensure that PC1 projection has a negative slope to reflect expected melting trend
        if linregress(range(self.loadings.shape[1]), self._transform[:,0]).slope > 0:
            self._transform = self._transform*-1
            self.loadings = self.loadings*-1
            self.pca.components_ = self.pca.components_*-1
        self.freqs = contact_df
        if significance_test == True:
            self.permutated_pca(N_permutations=N_permutations)
        else:
            self._permutated_explained_variance = None
            self.permutated_component_pvals = None
            self.chacra_pvals = None
            self.top_chacras = None

    def sorted_loadings(self, pc=1):
        '''
        Sort the original loadings in descending absolute value.

        Parameters
        ----------
        pc : int
            The pc/ chacra to sort the loading score data by.

        Returns
        -------
        pd.DataFrame
        '''
       
        return self.loadings.iloc[(-self.loadings['PC'+str(pc)].abs())
                                  .argsort()]
        
    def sorted_norm_loadings(self, pc=1):    
        '''
        Sort the normalized (positive 0 to 1) loading scores in descending
        order.

        Parameters
        ----------
        pc : int
            The pc/ chacra to sort the loading score data by.

        Returns
        -------
        pd.DataFrame
        '''
        return self.norm_loadings.iloc[(-self.norm_loadings['PC'+str(pc)]
                                                            .abs()).argsort()]
    
    def get_edges(self, pcs=None, inverse=True, as_dict=False):
        '''
        Generate networkx input for a network based on contact sensitivities.
        Network weights are top loading scores from pcs.

        TODO - original loading score sign is not considered which is probably
        imporant if networks based on contact sensitivies (rather than 
        probabilities) is going to be informative.
        
        Parameters
        ----------
        pcs : list
            List of integers.  If pcs = [1,2,3] then the edge weights will be 
            the loading score (normalized) from whichever pc the contact has 
            its highest score on.  
        
        inverse : bool
            Whether or not to return weights as the inverse of the loading score.
        
        as_dict : bool
            Return network input in dictionary form or list of tuples.

        Returns
        -------
        list or dict

        '''
        if pcs == None and self.top_chacras == None:
    
            print("Provide list of chacras from which to collect edge weights")
        
        elif pcs is not None:
            pcs = [f'PC{i}' for i in pcs]

        elif self.top_chacras is not None:
            pcs = [f'PC{i}' for i in self.top_chacras]

        # TODO - positive and negative loading scores?
        edges = []
        edge_dict = {}
        top_scores = self.norm_loadings[pcs].max(axis=1).values
        contacts = self.norm_loadings[pcs].max(axis=1).index
        for contact, score in zip(contacts,top_scores):
            partners = split_id(contact)
            if inverse == True:
                weight = 1/score
            else:
                weight = score
            edges.append((partners['resa'],partners['resb'],weight))

            
        if as_dict == True:
            for contact in edges:
                edge_dict[(contact[0],contact[1])] = contact[2]
            return edge_dict
        else:
            return edges  

                
    # def get_top_contact(self, resnum, pc_range=None):
    #     '''
    #     Return the contact name, normalized loading score, pc on which it has 
    #     its highest score, and the overall rank the score represents on the pc.
    #     pc_range is the range of PCs to include in the search for the
    #     highest score
        
    #     pc_range is inclusive
    #     #TODO this needs to be specific with chainid
    #     ## TODO Remove after confirming - just need get_top_score
    #     Get everything at once
    #     norm_loadings[['PC1','PC2','PC3']].max(axis=1)
    #     test_pca.norm_loadings[['PC1','PC2','PC3']].values.argmax(axis=1)+1
    #     '''
    #     if pc_range == None and self.top_chacras == None:
    #         pcs = ['PC'+str(i) for i in range(1,5)]
        
    #     elif pc_range is not None:
    #         pcs = ['PC'+str(i) for i in range(pc_range[0],pc_range[1]+1)]

    #     else:
    #         pcs = ['PC'+str(i) for i in self.top_chacras]


    #     contacts = []
    #     for contact in self.norm_loadings.index:
    #         #TODO add resname and/or chain 
    #         if str(resnum) in parse_id(contact).values():
    #             contacts.append(contact)


    #     highest_scores = self.norm_loadings[pcs].loc[contacts].max()
    #     top_score = highest_scores.sort_values()[-1]
    #     top_pc = highest_scores.sort_values().index[-1]
    #     contact = self.norm_loadings.loc[contacts][self.norm_loadings[pcs].loc[contacts][top_pc] == top_score].index[0]
    #     return {'contact':contact, 'PC':top_pc, 'loading_score':top_score}
    
    
    def get_top_score(self, contact, pc_range=None):

        '''
        Retrieve the contact's highest loading scores among the pcs in pc_range 
        (inclusive).

        contact : str
            The contact name.

        pc_range : list 
            List of integers corresponding to the PCs/ chacras that you 
            want the highest score from. If None and top_chacras attribute is
            available, the highest score among them will be returned. If neither
            is available, data for PCs 1-4 will be returned.

        Returns
        -------
        Dictionary with PC key and score value.

        If you want everything at once
        norm_loadings[['PC1','PC2','PC3']].max(axis=1)
        test_pca.norm_loadings[['PC1','PC2','PC3']].values.argmax(axis=1)+1
        '''
        if pc_range == None and self.top_chacras == None:
            pcs = ['PC'+str(i) for i in range(1,5)]
        
        elif pc_range is not None:
            pcs = ['PC'+str(i) for i in range(pc_range[0],pc_range[1]+1)]

        else:
            pcs = ['PC'+str(i) for i in self.top_chacras]

        data = {}
        vals = self.norm_loadings[pcs].loc[contact].values
        data[vals.argmax()+1] = vals.max()
        
        return data

 
    def get_chacra_center(self, pc, cutoff=0.6, absolute=True):
        '''
        Return the loading score dataframe containing only the contacts with 
        loading scores above the cutoff on specified pc.

        Parameters
        ----------
        pc : int
            The principal component from which to retrieve the most responsive 
            contacts.

        cutoff : float
            The minimum absolute value of loading score to include in the top 
            sensitive contacts.

        absolute : bool
            Whether to return the dataframe with normalized absolute values or 
            original loading scores. 
        
        Returns
        -------
        pd.DataFrame
        '''

    
        chacra_centers = self.sorted_norm_loadings(pc).loc[(self.sorted_norm_loadings(pc)[f'PC{pc}'] >= cutoff)].index
        if absolute == True:
            return self.sorted_norm_loadings(pc).loc[chacra_centers]
        else:
            return self.loadings.loc[chacra_centers]
        
    # def dynamic_energy_warping(self):
    #     '''
    #     Use dynamic time warping to find the explore contact frequency
    #     and pc projections
    #     '''
        
   

    def permutated_pca(self, N_permutations=500, get_loading_pvals=False):
        '''
        Randomize the values within the contact frequency columns to test the significance of the contact PCs.
        After this is run the suggested chacras for further investigation are available as .top_chacras.

        contact_frequencies : pd.DataFrame
            The dataframe of contact frequencies that the ContactPCA is based off of.

        N_permutations : int
            Number of times to randomize the dataframe and perform PCA.

        Returns
            np.array of explained variance by PC for each permutation of the dataframe.
        '''    
        # borrowed code from here https://www.kaggle.com/code/tiagotoledojr/a-primer-on-pca
        self._N_permutations = N_permutations
        df = self.freqs
        
       
        #original_variance = self.pca.explained_variance_ratio_
        pca = PCA()
        variance = np.zeros((N_permutations, len(df.index)))
        print('This can take a moment. Kshama.')
        for i in tqdm.tqdm(range(N_permutations)):
            X_aux = _de_correlate_df(df)    
            pca.fit(X_aux)
            # record the explained variance of this iteration's PCs
            variance[i, :] = pca.explained_variance_ratio_
            # track the behavior of the loading scores too
            # ....probably no value in it
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
    
        self.chacra_pvals = np.sum(np.abs(np.diff(variance, axis=1, prepend=0)) > \
                    np.abs(np.diff(self.pca.explained_variance_ratio_, prepend=0)), axis=0) / N_permutations

        deepest_chacra = np.where((self.chacra_pvals  > 0.05)==False)[0][-1] + 1
        self.top_chacras = list(range(1,deepest_chacra+1))


        