# -*- coding: utf-8 -*-
"""
Author: Dan Burns
"""
import pandas as pd
import numpy as np
import re
import pathlib
from sklearn.decomposition import PCA
from ChACRA.ContactAnalysis.contact_functions import _parse_id, check_distance_mda, _split_id
from scipy.stats import linregress
import MDAnalysis as mda
import collections
from ChACRA.ContactAnalysis.utils import *
from ChACRA.ContactAnalysis.contact_functions import *
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
    
    
    def contact_partners(self, resid1, resid2=None, id_only=False):
        '''
        filter the dataframe to only return contacts involving the provided residue(s)

        '''
        if resid2:
            regex1 = f"[A-Z1-9]+:[A-Z]+:{resid1}(?!\d)-[A-Z1-9]+:[A-Z]+:{resid2}(?!\d)"
            regex2 = f"[A-Z1-9]+:[A-Z]+:{resid2}(?!\d)-[A-Z1-9]+:[A-Z]+:{resid1}(?!\d)"
            regex = f"{regex1}|{regex2}"
        else:
            regex1 = f"[A-Z1-9]+:[A-Z]+:{resid1}(?!\d)-[A-Z1-9]+:[A-Z]+:\d+"
            regex2 = f"[A-Z1-9]+:[A-Z]+:\d+-[A-Z1-9]+:[A-Z]+:{resid1}(?!\d)"
            regex = f"{regex1}|{regex2}"
        return self.freqs.filter(regex=regex, axis=1)
    
      
    def all_edges(self, weights=True, inverse=True, temp=0, as_dict=False):
        '''
        returns list of contact id tuples for network analysis input
        inverse inverts the edge weight so something with a high contact
        frequency has a low edge weight and is treated as if it is 
        'closer' in network analysis.
        '''
        all_contacts = []
        for contact in self.freqs.columns:
            partners = _split_id(contact)
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
            partners = _split_id(contact)
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
            id_dict = _parse_id(contact)
            # check this 
            if id_dict['chaina'] != id_dict['chainb']:
                continue
            else:
                if np.sqrt((int(id_dict['resida'])
                            -int(id_dict['residb']))**2) > n_neighbors:
                    reduced_contacts.append(contact)
        return reduced_contacts
    
    


    def average_contacts(self, structure=None, identical_subunits=None, neighboring_subunits=None, opposing_subunits=False,
                         protein_only=True):
        '''
        oppposing subunits should let the user specify which subunits are opposite of each other (rather than adjacent)
        and during averaging, it will distinguish between adjacent and opposing and assign them to the right contact id
        i.e. A:res:num-C:res:num for oppsing and A....-B for adjacent

        # NOTE: Have to be careful if there are non protein molecules in the structure when averaging or else the identical_subunits
            list will not reflect what the protein subunits that you want to average.
            Supply the identical subunits in this case! e.g. ['A','B']
            # This should be taken care of now with the find_identical_subunits function

        neighboring_subunits: list of tuples
            for each list of identical subunits, you can specify which two of those should be used to name the averaged contact so they can be depicted 
            on the structure.  If None, the first two chain IDs from each identical subunit list will be used for the averaged contact naming.
            lists of neighboring subunits must be in same order as identical_subunits and must include a tuple of chain ids for each list in identical_subunits

        opposing_subunits: list of tuples
            Leave as False for dimers. This is intended for special cases like ion channels where it might be interesting to 
            specify contacts occurring between different pairs of identical subunits

            specify which subunits are opposite of one another (like in a tetrameric ion channel)
            if you want to separately track contacts that occur directly across an interface rather than
            contacts that occur on adjacent subunits.
            This should be left as None in most cases where it's not obvious and 
            when there are fewer than 4 subunits.
            example: opposing_subunits = [('B','D'),('A','C')]

        protein_only:  boolean
            If your structure includes anything other than protein but your contact information only deals with protein
        
        '''
        
        # original df
        print(opposing_subunits)
        odf = self.freqs.copy()

        if structure:
            u = mda.Universe(structure)
            if protein_only:
                protein = u.select_atoms('protein')
                u = mda.Merge(protein)

            if identical_subunits == None:
                # dictionary to deal with potentially more than one set of identical subunits 
                # mda segids picks up far right column of PDB seemingly if present - chainid option would be ideal
                #TODO right now what follows doesn't deal with contacts between non-identical subunits
                ## Could use the contact data to find which subunits interact with which other subunits
                ## Then use a regex with the possible hetero-chain combinations 
                identical_subunits = find_identical_subunits(u)
                for value in identical_subunits.values():
                    print(f'Using subunit IDs {" ".join(value)} for averaging.')
       
        averaged_data = {} 
        # dropped_columns = [] # not using atm
        # loop over each set of subunits in identical_subunits
        #TODO use a different dictionary that has already filtered the columns
        # then can jump straight to df = odf[contacts].copy()
        for z, subunits in enumerate(identical_subunits.values()):

            # put in alphabetical order 
            subunits.sort()
            # loop is using prefiltered dataframes containing chains involved in "subunits" 
            #TODO this will end up including duplicates when you have two or more lists
            ## if you don't remove the contacts from the original df
            contacts = get_all_subunit_contacts(subunits, odf)
          
            df = odf[contacts].copy()
            
            # create chain names for the new averaged contact name        
            if structure and opposing_subunits == None and neighboring_subunits == None:
                opposing_subunits = get_opposing_subunits(subunits, u)
                opposing_subunits.sort()
                chain1, chain2, alt_chain2 = opposing_subunits[0][0], opposing_subunits[1][0], opposing_subunits[1][1]
                print(opposing_subunits)
                # not using alt_chain2 atm but can use it when shortest contact is not AB but AC 
                # take first position
            if neighboring_subunits:
                chain1 = neighboring_subunits[z][0]
                chain2 = neighboring_subunits[z][1]
            else:
                chain1 = subunits[0]
                chain2 = subunits[1]
            # this will start a loop where after a column has been averaged,
            # the columns involved in the averaging will be dropped in place.
            # exit condition is after all columns in the filtered df have been dropped because they have been caught by the regular expression, 
            # sorted according to intra/ inter subunit condition, and averaged
            while len(df.columns) > 0:
                
                # picking up a new contact pattern
                # can check to see which identical_subunits list this falls into and 
                # adjust accordingly
                resids = _parse_id(df.columns[0])
                # intersubunit contacts can have swapped resids
                # so search with both regexes 
                regex1 = f"[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)-[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)"
                regex2 = f"[A-Z1-9]+:{resids['resnb']}:{resids['residb']}(?!\d)-[A-Z1-9]+:{resids['resna']}:{resids['resida']}(?!\d)"
                regex = f"{regex1}|{regex2}"
                to_average = list(df.filter(regex=regex, axis=1).columns)

                # If the contact is happening in the same subunit
                if resids['chaina'] ==  resids['chainb']:
                    # intra-subunit name format
                    contact = f"{chain1}:{resids['resna']}:{resids['resida']}-"\
                            f"{chain1}:{resids['resnb']}:{resids['residb']}"
                
                    # check to make sure none of the contacts in to_average have different chain IDs
                    # and remove them from to_average - they'll get picked up in the else block
                    to_remove = filter_by_chain(to_average, same_chain=False)
                    for pair in to_remove:
                        to_average.remove(pair)
                    # record the averaged contact
                    averaged_data[contact] = get_standard_average(df, to_average, subunits)
                    # and then remove the original contacts from the df
                    df.drop(to_average, axis=1, inplace=True)
                    # odf.drop(to_average, axis=1, inplace=True) # can just 
                    #dropped_columns.extend(to_average)

                else:
                    # Average contacts that are occuring inter-chain
                    contact = f"{chain1}:{resids['resna']}:{resids['resida']}-"\
                            f"{chain2}:{resids['resnb']}:{resids['residb']}"
                    # moved to average from here
                    # Return contacts that have the same chain ids (same_chain=True)
                    to_remove = filter_by_chain(to_average, same_chain=True)               
                    for pair in list(set(to_remove)):
                        to_average.remove(pair)
        
                    if structure:
                        # this tries to ensure the correct depiction if visualized on the structure using functions in contacts_to_pymol
                        #  TODO alt_chain2 is not consistent with other options of naming subunits
                        contact = check_distance_mda(contact,u, chain1, chain2)     

                    ################## catch opposing subunits 
                    # treating opposing subunits differently might be useful. 
                    if opposing_subunits == False:
                        pass
                    else:
                        # (Unnecessary for dimers)
                        opposing_contacts = get_opposing_subunit_contacts(to_average, opposing_subunits)

                        for pair in opposing_contacts:
                            # remove it from main record
                            to_average.remove(pair)
                        # get the mean
                        if len(opposing_contacts) > 0:
                            opposing_contact_name = f"{opposing_subunits[0][0]}:{resids['resna']}:{resids['resida']}-"\
                                                    f"{opposing_subunits[0][1]}:{resids['resnb']}:{resids['residb']}"
                            averaged_data[opposing_contact_name] = df[opposing_contacts].mean(axis=1)
                        # done with these contacts
                        df.drop(opposing_contacts, axis=1, inplace=True)
                        #dropped_columns.extend(opposing_contacts)
                    ########################################################################################
                    # get the average.
                    averaged_data[contact] = get_standard_average(df,to_average, subunits, check=False)
                    # done with this iteration
                    df.drop(to_average, axis=1, inplace=True)
                    #dropped_columns.extend(to_average)
            return pd.DataFrame(averaged_data)



    def renumber_residues(self, starting_residue_number):   
        '''renumber the residues so the first residue begins with
        starting_residue_number.  Useful if the contact_files generated with
        get contacts was made with a incorrectly numbered structure file starting
        from 1.
        '''

        # TODO add option to renumber from several starting points/chains
        mapper = {}
        for column in self.freqs.columns:
            split_ids = _parse_id(column)
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
            resinfo = _parse_id(contact)

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
            resinfo = _parse_id(contact)
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
    
    X_aux = df.copy()
    for col in df.columns:
        X_aux[col] = df[col].sample(len(df)).values
        
    return X_aux

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
            partners = _split_id(contact)
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
            partners = _split_id(contact)
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
            if str(resnum) in _parse_id(contact).values():
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