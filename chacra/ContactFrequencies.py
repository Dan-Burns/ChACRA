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
import tqdm
from scipy.stats import linregress 
from ChACRA.chacra.average import everything_from_averaged
from ChACRA.chacra.visualize.contacts_to_pymol import \
    pymol_averaged_chacras_to_all_subunits, get_contact_data, to_pymol
from ChACRA.chacra.utils import multi_intersection
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
    
    
    
    def __init__(self, contact_data, 
                 temps=None, 
                 temp_progression=None, 
                 min_max_temp=None, 
                 structure=None,
                 get_chacras=True, 
                 N_permutations=500,
                 verbose=False):
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
            Conveniently makes the structure available to other functions. 

        get_chacras : bool
            Make the ContactPCA class an attribute of this object.  
            i.e. cont = ContactFrequencies(data,get_chacras=True)
                cont.cpca.get_chacra_centers(1)...
        
        N_permutations : int
            If get_chacras == True, the number of times to permutate the data to
            obtain chacra (PC) significance values.
        
        verbose : bool
            For debugging.

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
        
                    print('Provide a file with a .csv or .pd (pickle format) '\
                          'file extension.\n'\
                            'Or provide a path the folder containing the '\
                                'original getcontacts .tsv files.')
            
            elif os.path.isdir(contact_data):

                contact_files = [f'{contact_data}/{file}' for file in sorted(
                            os.listdir(contact_data),key=lambda x: int(
                                                    re.split(r'_|\.',x)[-2]))
                              if file.endswith('.tsv')]
                
                if verbose == True:
                    for file in contact_files:
                        print(file, flush=True)
                contact_dictionary = make_contact_frequency_dictionary(
                                                            contact_files)
                self.freqs = pd.DataFrame(contact_dictionary)
        except:
            try:
                if type(contact_data) == pd.DataFrame:
                    self.freqs = contact_data
                elif type(contact_data) == dict:
                    self.freqs = pd.DataFrame(contact_data)
            except:
                ("Provide one of file (.pd or .csv extension), pd.DataFrame, "\
                 "dict, or path to original getcontacts .tsv frequency files.\n")
            
        
            
        if temps is not None:
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

        # give access to the ContactPCA 
        if get_chacras == True:
            self.cpca = ContactPCA(self.freqs, 
                                   N_permutations=N_permutations,
                                   structure=structure)
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
            regex1 = rf"{chain}:{resn}:{resid1}(?!\d)-{chain2}:{resn2}:{resid2}(?!\d)"
            regex2 = rf"{chain2}:{resn2}:{resid2}(?!\d)-{chain}:{resn}:{resid1}(?!\d)"
            regex = rf"{regex1}|{regex2}"
        else:
            if type(resid1) == tuple:
                chain = resid1[0]
                resn = resid1[1]
                resid1 = resid1[2]
            else:
                chain = '[A-Z1-9]+'
                resn = '[A-Z]+'
            regex1 = rf"{chain}:{resn}:{resid1}(?!\d)-[A-Z1-9]+:[A-Z]+:\d+"
            regex2 = rf"[A-Z1-9]+:[A-Z]+:\d+-{chain}:{resn}:{resid1}(?!\d)"
            regex = rf"{regex1}|{regex2}"
        return self.freqs.filter(regex=regex, axis=1)
    
    
    #TODO function to identify the most inversely correlated contact involving each 
    # partner in a given contact

    def find_correlated_contacts(self, contact, inverse=True):
        '''
        Returns a list of n_contacts involving one member of the input contact
        with the highest (inverse) correlation values.
        This helps identify a contact that is made as a function of another one
        breaking.

        Parameters
        ----------
        contact : str
            The contact name to find correlated contacts for.

        inverse : bool  
            If True, return contacts with the highest inverse correlation values.

        Returns
        -------
        list
        List of contacts sorted in descending order of correlation values.
        '''
        # TODO a lower temperature range might correlate with one contact
        # while a higher temperature range might begin correlating with another
        # this might work best using a linear combination of PCs to identify
        # patterns of contact making and breaking that depend on more than one contact

        # can also return based on the whole matrix from 
        # self.freqs.corr().filter(regex=rf'{a}|{b}',axis=0).filter(regex=rf'{a}|{b}',axis=1).min()...
        

        a,b = contact.split('-')
        cor = self.freqs.corr()[[contact]]
        if inverse == True:
            out = cor.filter(regex=rf'{a}|{b}',axis=0).sort_values(by=contact,ascending=True)
        else:
            out = cor.filter(regex=rf'{a}|{b}',axis=0).sort_values(by=contact,ascending=False)
        return out.index.tolist()
        


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


    def to_heatmap(self,output_format='mean', pc=None, row=None):

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
         
        output_format : str
            format options are 'mean', 'stdev', 'difference', 'loading_score',
            of 'frequency'.
            If 'difference', specify tuple of rows your interested in taking the 
            difference from.
            If 'loading_score', then specify contact pc/chacra from which you 
            want the loading score. cpca attribute must be available.
        pc : int
            If output_format == loading_score, provide the principal component
            number for which loading scores should be heatmapped.
        row : int or float
            If output_format == frequency, provide the temperature or row
            corresponding to the frequency data to heatmap.

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
       
        # vectorize the data
        df_array = self.freqs.values
        loading_array = self.cpca.loadings.values
        # get the row index if format is 'frequency' 
        if row is not None:
            if row in self.freqs.index:
                # if it's provided as a temp, convert to row index
                row = np.where(self.freqs.index == row)[0][0]
            elif row in range(self.freqs.shape[0]):
                row = row
            else:
                print(f"Row {row} is not in the frequency data index. "\
                      f"Specify an integer row index instead if "\
                      f"{row} was a temperature value.")
        # get the indices for the residues in each contact to fill in the 
        # 'data' np.array
        # enumerate so you can reference the np.array form of the original data
        # faster.

        for i,contact in enumerate(self.freqs.columns):

            
            resinfo = parse_id(contact) 
            index1 = all_resis.index(f"{resinfo['chaina']}{resinfo['resida']}")
            index2 = all_resis.index(f"{resinfo['chainb']}{resinfo['residb']}")
            

            values = {}                               
            values['mean']  = df_array[:,i].mean(axis=0) 
            values['stdev'] = df_array[:,i].mean(axis=0) 
            values['difference'] = np.abs(df_array[-1,i] - df_array[0,i]) 
            
            if hasattr(self, "cpca") and pc is not None:
                #TODO offer sorted loadings to catch sign
                values['loading_score'] = loading_array[i,pc-1] 
            elif pc is not None and hasattr(self, "cpca") == False:
                print("Instantiate the cpca attribute with ContactPCA.")
                break
            if row is not None:
                values['frequency'] = df_array[row,i]

            data[index1][index2] = values[output_format]
            data[index2][index1] = values[output_format]
        
        return pd.DataFrame(data, columns=all_resis, index=all_resis)       


def de_correlate_df(df):
    '''
    randomize the rows within a dataframe's columns
    '''
   
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

    #TODO exclude contact (or regions) that have slopes that exceed the PC1
    slope at high temperature (indicating major melting)
    '''
    
    
    def __init__(self, 
                 contact_df, 
                 significance_test=True, 
                 N_permutations=500,
                 structure=None):
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

        # ensure that PC1 projection has a negative slope to reflect its  
        # expected melting trend
        if linregress(range(self.loadings.shape[1]), 
                      self._transform[:,0]).slope > 0:
            self._transform = self._transform*-1
            self.loadings = self.loadings*-1
            self.pca.components_ = self.pca.components_*-1
        self.freqs = contact_df
        if significance_test == True:
            # TODO Multiprocess
            self.permutated_pca(N_permutations=N_permutations)
            self.score_sums = self.get_score_sums()
        else:
            self._permutated_explained_variance = None
            self.permutated_component_pvals = None
            self.chacra_pvals = None
            self.top_chacras = None
            self.score_sums = None
        if structure is not None:
            self.structure = structure
        self.freqs = contact_df

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
    
    
    def get_top_score(self, contact, pc_range=None):

        '''
        Retrieve the contact's highest loading scores among the pcs in pc_range 
        (inclusive).

        contact : str
            The contact name.

        pc_range : tuple of int 
            #TODO this breaks easy.  Has to be 1 and max pc you want results from
            # Or else argmax() will return the wrong PC
            #TODO return as tuple instead of dictionary
            
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

    
        chacra_centers = self.sorted_norm_loadings(pc).loc[
            (self.sorted_norm_loadings(pc)[f'PC{pc}'] >= cutoff)].index
        if absolute == True:
            return self.sorted_norm_loadings(pc).loc[chacra_centers]
        else:
            return self.loadings.loc[chacra_centers]
   
    def permutated_pca(self, N_permutations=500, get_loading_pvals=False):
        '''
        Randomize the order of the contact frequency rows in each column 
        ndependently and perform PCA on the scrambled data to test the 
        significance of the contact PCs. The probability that the difference in 
        adjacent eigenvalues is greater in the scrambled data than the original
        data serves as a P value for each PC.
        After this is run the suggested chacras for further investigation are 
        available as .top_chacras.

        contact_frequencies : pd.DataFrame
            The dataframe of contact frequencies that the ContactPCA is based off of.

        N_permutations : int
            Number of times to randomize the dataframe and perform PCA.

        Returns
            np.array of explained variance by PC for each permutation of the dataframe.

        TODO multiprocess - divide N_permutations by cores
        '''    
        # adapted from here https://www.kaggle.com/code/tiagotoledojr/a-primer-on-pca
        self._N_permutations = N_permutations
        df = self.freqs
        
       
        pca = PCA()
        variance = np.zeros((N_permutations, len(df.index)))
        print('This can take a moment. Kshama.')
        for i in tqdm.tqdm(range(N_permutations)):
            
            X_aux = de_correlate_df(df)    
            pca.fit(X_aux)
            # record the explained variance of this iteration's PCs
            variance[i, :] = pca.explained_variance_ratio_
            # track the behavior of the loading scores too
            # ....probably no value in it
            if get_loading_pvals:
                if i == 0:
                    permutated_components = (np.abs(pca.components_) >= \
                                             np.abs(self.pca.components_))*1
                else:
                    # summing up the number of times that the randomized data loadings have greater values than the 
                    # real data
                    permutated_components += (np.abs(pca.components_) >= \
                                              np.abs(self.pca.components_))*1
        self._permutated_explained_variance = variance
        # The average of this tells you the probability of randomized data having higher loading scores
        # if this value is low (<0.05) then the real loading score is significant
        if get_loading_pvals:
            self.permutated_component_pvals = permutated_components/N_permutations
    
        self.chacra_pvals = np.sum(np.abs(
                                    np.diff(variance, axis=1, prepend=0)) > \
                    np.abs(np.diff(
                        self.pca.explained_variance_ratio_, prepend=0)
                        ), axis=0) / N_permutations

        deepest_chacra = np.where((self.chacra_pvals  > 0.05)==False)[0][-1] + 1
        self.top_chacras = list(range(1,deepest_chacra+1))

    def get_score_sums(self):
        '''
        For each residue, assign half the value of every loading score associated
        with it and sum (absolute values) them all.  This gives an idea of how
        much of the overall variance of a PC a residue is responsible for.

        Returns
        -------
        pd.DataFrame

        Columns are residue IDs and rows are principal component IDs.

        '''
        results = {pc:{} for pc in self.top_chacras}
        for col in self.norm_loadings.index:
            a, b = col.split("-")
            if a not in results[1].keys():
                for pc in self.top_chacras:
                    results[pc][a] = self.norm_loadings[f'PC{pc}'].loc[col]/2
            else:
                for pc in self.top_chacras:
                    results[pc][a] += self.norm_loadings[f'PC{pc}'].loc[col]/2
                
            if b not in results[1].keys():
                for pc in self.top_chacras:
                    results[pc][b] = self.norm_loadings[f'PC{pc}'].loc[col]/2
            else:
                for pc in self.top_chacras:
                    results[pc][b] += self.norm_loadings[f'PC{pc}'].loc[col]/2
        
        results = sort_nested_dict(results)
        
        return pd.DataFrame([result.values() for result in results.values()], 
                            index=results.keys(), columns=results[1].keys())

    def to_pymol(self, pcs=None, cutoff=0.6, 
                 output='chacra_selections.pml',
                 group_pcs=True,
                 reconstruct_from_averaged=False,
                 original_contacts=None,
                 representative_chains=None):
        '''
        Write a .pml file to visualize chacras on the structure.

        Parameters
        ----------
        pcs : list
            List of integer values corresponding to the PCs/ chacras you want
            to visualize.

        cutoff : float
            The minimum normalized loading score of the contacts to include
            from pcs.

        output : str
            The filepath/name of output file.

        #####################################################################
        Remaining arguments are only necessary if you are using averaged data
        AND want to depict the averaged chacras on all the subunits.
                            #########################
        
        reconstruct_from_averaged : bool
            If the data is averaged from a multimer, display the chacra contacts
            on all subunits. Must provide the original (before averaging) contact
            dataframe to 'original_contacts'.

        original_contacts : pd.DataFrame
            The unaveraged, original contact data.  Only required if 
            reconstruct_from_averaged=True.

        representative_chains : list
            The chain IDs that were used as the representative chains for the
            averaged data. 

        TODO: Need a function to reconstruct data from averaged with fewer
        required arguments.
        '''
        if pcs is not None:
            pass
        elif pcs is None and self.top_chacras is not None:
            pcs = self.top_chacras
        else:
            print("Provide a list of chacras to visualize with the 'pcs' arg.")
        
        if reconstruct_from_averaged is True and self.structure is None:
            print("You must assign a structure file to cpca.structure before "\
                  "the contacts can be reconstructed on all subunits.")
        
        elif reconstruct_from_averaged is True and original_contacts is None:
            print("Provide the original (all contacts before averaging) "\
                  "contacts data if you want to reconstruct the full structure."\
                  )
        elif reconstruct_from_averaged is True and representative_chains is None:
            print("Provide a list of the representative chain ids use for "\
                  "averaging.")

        top_contacts = []
        for i in pcs:
            top_contacts.extend(list(self.get_chacra_center(i,cutoff).index))
            # top_contacts.extend(self.sorted_norm_loadings(i).loc[
            #                     self.sorted_norm_loadings(i)[f'PC{i}'] > cutoff
            #                                                     ].index)
        top_contacts = list(set(top_contacts))

        if reconstruct_from_averaged is True:
            u = mda.Universe(self.structure)
            mapped_contacts = everything_from_averaged(self.freqs[top_contacts], 
                                                   original_contacts, 
                                                   u, 
                                                   representative_chains,
                                                   as_map=True)
            pymol_data = get_contact_data(
                mapped_contacts.keys(),
                self.freqs,
                self, 
                pc_range=(pcs[0],pcs[-11])) # pc_range should be changed to list
            pymol_averaged_chacras_to_all_subunits(mapped_contacts,
                                                    pymol_data, 
                                                    output)

        else:
            cont = ContactFrequencies(self.freqs,
                                        get_chacras=False)
            to_pymol(top_contacts, 
                     cont,
                     self, 
                     output, 
                     pc_range=(pcs[0],pcs[-1]), 
                     group=group_pcs)


#TODO class for combined chacra 
# hold all the original data ContactFrequencies and ContactPCA
# hold a combined contact pca with appened system names to all contacts
# hold a difference ContactFrequencies with the original names
    







class CombinedChacra():
    '''
    Combine multiple contact frequency data sets and perform PCA on them.
    This places the contacts from multiple ensembles on the same axis system for
    comparitive analysis.

    Most of the functionality here is only applicable to pairs of ensembles
    so restrict CombinedChacra data_dict input to length 2. In the future 
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with keys (strings) defining the name of the corresponding
        pd.DataFrame values.  The dataframes must all contain the same number of 
        rows (temperatures).
        Example
        -------
        {'apo':pd.DataFrame(apo_contact_frequencies),
        'holo': pd.DataFrame(holo_contact_frequencies)}
        
    
    '''
    def __init__(self, data_dict):

        self.original_data = data_dict
        self.names = list(data_dict.keys())
        # prepend each datasets key (name) to each contact id
        # eg. {'apo', contact_df} will create contacts from contact_df with names
        # in the form of 'apo_A:LYS:100-A:ASP:110'
        self.mappers = {key:{col:f'{key}_{col}' for col in data_dict[key].columns} 
                for key in data_dict.keys()}
        
        self.reverse_mappers = {
                                name:{val:key for key, val 
                                      in self.mappers[name].items()}
                                for name in self.mappers
        }
        
        self.shapes = {key:df.shape for key, df in data_dict.items()}

        self.shared_contacts = list(multi_intersection([(list(data.keys())) for data 
                                                   in self.mappers.values()]))
        self.not_shared_contacts = {
                    key:[contact for contact in self.mappers[key].keys() 
                         if contact not in self.shared_contacts] 
                         for key in self.mappers.keys()
                           }
        
        
        self.combined_freqs = pd.concat(
                            [data_dict[key].copy().rename(self.mappers[key], axis=1) 
                            for key in data_dict],
                                        axis=1)
    
        print("Getting chacras of the combined data.")
        self.combined = ContactFrequencies(self.combined_freqs)
        self.separate_ensemble_loadings()
     
        # create difference data
        difference_data = {}
       
        # Right now this is only going to handle data involving 2 ensembles
        # will have to use something like the combined_freqs prepend scheme
        # for differences between more than 2 ensembles
        for contact in self.shared_contacts:
            diff = np.abs(self.original_data[self.names[0]][contact].values - 
                    self.original_data[self.names[1]][contact].values)
            difference_data[contact] = diff
        for name in self.not_shared_contacts:
            for contact in self.not_shared_contacts[name]:
                difference_data[contact] = self.original_data[name][contact].values
        print('Getting the chacras of the contact frequency differences.')
        self.differences = ContactFrequencies(pd.DataFrame(difference_data))
        
    def get_top_changes(self, cutoff, min_loading_dif=.2, pc_range=None):
        '''
        Get the contacts from the combined chacras that are present above the 
        loading score cutoff in one ensemble but not the other

        Parameters
        ----------

        cutoff: float
            Minimum normalized (absolute) loading score to consider. If a 
            contact is above this in one ensemble and below in another on the 
            same pc this contact will be reported.

        pc_range: tuple of int
            The range of pcs to consider (inclusive).

        Returns
        -------
        Dictionary
        The pcs (keys) and lists tuples of contacts with prepended names that are 
        in the top contacts on the given pc when the other ensemble's identical
        contact is not.
        The first tuple item is in the top, the second isn't.
        '''
        
        if pc_range is None and self.combined.cpca.top_chacras is not None:
            pcs = self.combined.cpca.top_chacras
        elif pc_range is not None:
            pcs = list(range(pc_range[0],pc_range[1]+1))
        
        different = {pc:[] for pc in pcs}

        for pc in pcs:
            above_cutoff = self.combined.cpca.norm_loadings.loc[
                        self.combined.cpca.norm_loadings[f'PC{pc}'] > cutoff
                ].index
            
            for contact in self.shared_contacts:
                contacta = f'{self.names[0]}_{contact}'
                contactb = f'{self.names[1]}_{contact}'
                if (contacta in above_cutoff and 
                contactb not in above_cutoff) \
                    and ((self.combined.cpca.norm_loadings[f'PC{pc}'].loc[
                                                                    contacta]
                    - self.combined.cpca.norm_loadings[f'PC{pc}'].loc[
                                                                    contactb])
                        > min_loading_dif):
                    different[pc].append((contacta, 
                                         contactb))
                    
                elif (contactb in above_cutoff and 
                contacta not in above_cutoff) \
                    and ((self.combined.cpca.norm_loadings[f'PC{pc}'].loc[
                        contactb]
                    - self.combined.cpca.norm_loadings[f'PC{pc}'].loc[contacta])
                        > min_loading_dif):
                    different[pc].append((contactb, 
                                         contacta))

        return different



    def get_flipped_contacts(self, cutoff, pc_range=None,
                             plot=False):
        '''
        Get contacts that have flipped loading scores on the same combined PC.

        TODO - this is slow

        Parameters
        ----------
        cutoff : float
            Only look for pairs of contacts where at least one of the contacts
            has a loading score above cutoff. This means you're just looking
            for flipped signs among the energy-sensitive contacts.

        pc_range : tuple of int
            The range of pcs to restrict the search to (inclusive).
            If None, combined.top_chacras will be used.

        Returns
        -------
        List of contacts that have flipped loading scores on the shared principal
        components.
        '''
        print("This one takes a moment...")
        if pc_range is None:
            pcs = self.combined.cpca.top_chacras
        else:
            pcs = list(range(pc_range[0], pc_range[1]+1))

        different_signs = set()
        for pc in pcs:
            for contact in self.shared_contacts:
                # if they're opposite signs
                contacta = f'{self.names[0]}_{contact}'
                contactb = f'{self.names[1]}_{contact}'
                if (self.combined.cpca.loadings[f'PC{pc}'][contacta] > 0 and                                                     
                    self.combined.cpca.loadings[f'PC{pc}'][contactb] < 0) or\
                    (self.combined.cpca.loadings[f'PC{pc}'][contacta] < 0 and                                                       
                    self.combined.cpca.loadings[f'PC{pc}'][contactb] > 0):                                                          
                    # and at least one is above the cutoff
                    if (self.combined.cpca.sorted_norm_loadings(pc)[
                        f'PC{pc}'].loc[contacta] < cutoff) and \
                        (self.combined.cpca.sorted_norm_loadings(pc)[
                            f'PC{pc}'].loc[contactb] < cutoff):
                        pass
                    # and at least one has its highest score on the current pc
                    elif (list(self.combined.cpca.get_top_score(contacta).keys()
                               )[0] == pc) or \
                        (list(self.combined.cpca.get_top_score(contactb).keys()
                              )[0] == pc):
                        different_signs.add(contact)
        
        return list(different_signs)
                
    def get_changes(self, stdev_min=0.0, stdev_max=0.02, mean_dif=0.2):
        '''
        Identify contacts with significantly different frequencies between the 
        two ensembles based on each contact's standard deviation and mean.

        # TODO as is, this is useful for finding two flatline contacts
        # That have noticeable differences
        # Could offer cutoff criteria for each member to identify when 
        # on is sensitive and the other isn't (slopes, stdevs, etc.)

        Parameters
        ----------
        stdev_min: float
            The minimum standard deviation for the contact.
        stdev_max: float
            The maximum standard deviation for the contact. Low max standard
            deviation limits the search to stable contacts.
        mean_dif: float
            The minimum difference between the means of the two contact pairs 
            that will be reported on.

        Returns
        -------
        List of contacts (which occur in both ensembles)with changes in contact
        frequency behavior between two ensembles that meet the input criteria.
        '''
        different= []

        for contact in self.shared_contacts:
            if (((stdev_min < self.original_data[self.names[0]][contact
                                                            ].std() < stdev_max) 
                 and (stdev_min < self.original_data[self.names[1]][contact
                                                    ].std() < stdev_max)) and 
                (np.abs(self.original_data[self.names[0]][contact].mean() - \
                self.original_data[self.names[1]][contact].mean()) > mean_dif)):
                different.append(contact)

        return different
    
    def get_real_unique_contacts(self, cutoff=0.05, criteria='mean'):
        '''
        Contacts that only occur in one ensemble or the other and have
        a mean value above the cutoff will be returned. 
        This is useful because contacts that only occur in one ensemble at very
        low frequency might be considered noise, while ones that have more 
        significant values might be the result of a bound effector.

        cutoff: float
            The minimum frequency that a contact must exceed to be returned.

        criteria : str
            'mean' or 'max'

        Returns
        -------
        Dictionary of lists of contacts that exceed the cutoff.
        '''
        real_contacts = {name: None for name in self.names}
        
        for name in self.not_shared_contacts:
            if criteria == 'mean':
                mask = self.original_data[name][self.not_shared_contacts[name]].mean()\
                  > cutoff
            elif criteria == 'max':
                mask = self.original_data[name][self.not_shared_contacts[name]].max()\
                  > cutoff
            real_contacts[name] = list(mask.index[mask])
            

        return real_contacts
    
    def separate_ensemble_loadings(self):
        '''
        Take a combined chacra dataframes and separate them into ContactPCA objects.
        Provides access to the sorting methods and can be used in the pymol
        visualization functions.
        
        Paramters
        ---------
        names: list
            The list of names that have been prepended in to the combined 
            contact frequency names.
        
        df: pd.DataFrame
            The combined loading score or contact frequency df
            
        Returns
        -------
        Dictionary
        Names are keys and dataframe values with prepended names removed from 
        the contact ids
        '''
        # only really need this for loading score df
        if self.combined.cpca.loadings.columns[0] == "PC1": # loading score df
            id_axis = 0 
        else:
            id_axis = 1


        self.separated_cpca = {name:ContactPCA(self.original_data[name], 
                                          significance_test=False) for name in 
                                          self.names
                                          }
        for name in self.names:
            # The transform and pca object won't correspond to the 
            # separated loadings.  
            # TODO fix this so you can project each ensemble onto the shared
            # principal components
            self.separated_cpca[name].pca = None
            self.separated_cpca[name]._transform = None
            self.separated_cpca[name].loadings = \
                self.combined.cpca.loadings.filter(like=name, axis=id_axis).rename(
                                        self.reverse_mappers[name], axis=0
                                                 )
            # normalized values are distributed between the ensembles
            # so only one ensemble will have a maximum of 1 on a given pc
            self.separated_cpca[name].norm_loadings = \
                                    self.separated_cpca[name].loadings
                                                                

    
