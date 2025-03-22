## Networkx functions


# network visualization
# https://robert-haas.github.io/gravis-docs/
# construct minimal graph that connects all the most sensitive contacts
import networkx as nx
import itertools
from networkx.algorithms import community
from networkx import edge_betweenness_centrality as betweenness
from .utils import sort_dictionary_values

def edge_to_contact(edge_data,contact_data):
    '''
    Convert networkx edge back to original contact name.
    edge : tuple or list of tuples
        networkx formatted contact name or list of them.
    
    contact_data : ContactFrequencies or ContactPCA


    '''
    if hasattr(contact_data, 'freqs'):
        contacts = contact_data.freqs.columns
    elif hasattr(contact_data, 'loadings'):
        contacts = contact_data.loadings.index
    else:
        print("You must provide a ContactFrequencies or ContactPCA object.")
    
    if type(edge_data) == list:
        contact_list = []
        for edge in edge_data:
            resa,resb = edge[0], edge[1]
            contact =  f'{resa}-{resb}'
            if contact not in contacts:
                contact = f'{resb}-{resa}'
                if contact not in contacts:
                    print(f"can't find {resa}-{resb} or {resb}-{resa}")
                    
            contact_list.append(contact)
        return contact_list
    else:
        resa,resb = edge_data[0], edge_data[1]
        contact =  f'{resa}-{resb}'
        if contact not in contacts:
            contact = f'{resb}-{resa}'
            if contact not in contacts:
                print(f"can't find {resa}-{resb} or {resb}-{resa}")
                return

        return contact

def make_network(cont, 
                 temp, 
                 
                 ):
    '''

    cont : ContactFrequencies
        The ContactFrequencies object containing the contacts 
        to compose the network from.
    
    temp : the temperature corresponding to the data row to take your contact
    
    selection : list
        list of selected contacts to construct the graph from.

    Returns
    -------
    nx.Graph
    '''
    # you want the edge weights to be inverse contact frequencies
    # lower edge weight value means "closer"/ higher contact frequency
    #TODO add exclusion cutoffs
   
    inverse = cont.get_edges(temp=temp, as_dict=True)
    original = cont.get_edges(temp=temp, inverse=False)

    G = nx.Graph()

    G.add_weighted_edges_from(original)
    nx.set_edge_attributes(G,inverse,name='inverse')
    return G



def get_communities(G, n_communities=5, removed_edges=False):
    '''
    # visualize with
    for i, community in enumerate(communities):
        nx_to_pymol(f'{out_dir}/pymol_selections/networkx_selections/6_communities_27C.pml', '6_communties_temp_7',community,i)

    
    '''
    if removed_edges == True:
        removed_edges = {}
        def most_central_edge(G):
            '''
            Use in girvan_newman community analysis to identify the removed edge on each iteration
        
            Example
            -------
            removed_edges = {}
            comp = community.girvan_newman(network_obj, most_valuable_edge=most_central_edge)
            # choose number of communities
            k = 6
            limited = itertools.takewhile(lambda c: len(c) <= k, comp)
            # pull the community lists out with itertools 'takewhile' 
            for communities in limited:
                print(tuple(sorted(c) for c in communities))
            
            '''
            centrality = betweenness(G, weight="inverse")
            edge = max(centrality, key=centrality.get)
            removed_edges[edge] = centrality[edge]
            return edge
        comp = community.girvan_newman(G, most_valuable_edge=most_central_edge)
        
    else:
        comp = community.girvan_newman(G)
    # choose number of communities
    k = n_communities
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    # pull the community lists out with itertools 'takewhile' 
    stored_communities = {}
    for i, communities in enumerate(limited):
        stored_communities[i] = tuple(sorted(c) for c in communities)
    if removed_edges is True:

        return communities, removed_edges
    else:
        return communities

def get_betweenness_centrality(G, weight='inverse'):
    '''
    If using on averaged data, consider reconstructing the whole complex's network with average.everything_from_averaged
    before investigating centrality.
    '''

    #TODO check weight argument as None

    betweeness = nx.edge_betweenness_centrality(G, weight=weight)
    sorted_edges = sort_dictionary_values(betweeness)
    
    return sorted_edges

def make_minimum_graph(contact_data, return_contacts=True):
    '''
    Not Implemented

    Construct the minimal graph that connects the top chacra contacts
   
    Parameters
    ----------
    contact_data : ContactFrequencies or ContactPCA
        calls .get_edges on whichever object is provided

    return_contacts : bool
        If True, return a list of contacts. If False, return the networkx Graph.
    '''

    #TODO option to specify which chacras in arguments
    # https://en.wikipedia.org/wiki/Steiner_tree_problem
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.mst.minimum_spanning_tree.html
    # cycle graph....
    edges = contact_data.get_edges()
    G = nx.Graph()
    #TODO add inverse or original weights
    #TODO make_network should handle contact frequencies or loading scores
    G.add_weighted_edges_from(edges)
    min_graph = nx.minimum_spanning_tree(G)
    if return_contacts == True:
        return edge_to_contact(list(min_graph.edges), contact_data)
    else:
        return min_graph
    
def pc_network(start_contact, contact_data, n_contacts=100, end_contact=None,
               max_pc=None):
    '''
    Given a starting contact, find the next contact that involves either residue that has the highest
    loading score on any of the top principal components and continue until you reach n_contacts or a specified contact.

    Parameters
    ----------
    start_contact : str
        The contact to start from.

    contact_data : ContactPCA
        The ContactPCA object containing the PCA data.

    n_contacts : int
        The number of contacts to return.
        
    end_contact : str
        The contact to end at.
        If None, the function will continue until n_contacts is reached.
        If a contact is specified, the function will stop when that contact is reached.
    
    max_pc : int
        The maximum principal component to consider.
        If None, top_chacras will be considered.

    Returns
    -------
    list
    The list of connected contacts.
    '''
    # start with contact and working above the loading score cutoff
    # find the next contact that involves either residue that has the highest
    # loading score on any of the top principal components
    # and continue until you reach n_contacts or a specified contact
    # return the network of these contacts
    # TODO: incorparate splitting of paths from a contact
    # this is all probably done best using edge betweeness with loading scores as weights
    path = [start_contact]
    next_contact = path[-1]
    while (len(path) < n_contacts) and (end_contact is None or end_contact not in path):

        resa, resb = next_contact.split("-")
        # all the columns that involve either residue
        cols = list(contact_data.norm_loadings.filter(regex=f'{resa}|{resb}',axis=0).index)
        # only want residues to appear two consecutive times.
        counts = {}
        for previous in path:
            # if the contact is already in the path, don't include them in the 
            # list of contacts to consider
            if (previous in cols):
                cols.remove(previous)
            # if a residue is in the path twice, don't include it in the list of contacts to consider
            a, b = previous.split("-")
            if a in counts.keys():
                counts[a]+=1
            else:
                counts[a] = 1
            if b in counts.keys():
                counts[b]+=1
            else:
                counts[b] = 1
        # only want residues to appear two consecutive times.
        counts = {res:count for res, count in counts.items() if count >= 2}
        # remove those that have here
        to_remove = []
        for res in counts:
            for col in cols:
                if res in col:
                    to_remove.append(col)
        for col in set(to_remove):
            cols.remove(col)
            
        if max_pc is not None:
            dat = contact_data.norm_loadings.loc[cols][
            [f"PC{pc}" for pc in range(1, max_pc+1)]]
        else:
            dat = contact_data.norm_loadings.loc[cols][
            [f"PC{pc}" for pc in contact_data.top_chacras]] 
        if dat.empty:
            break  
        next_contact = dat.idxmax().max()
        path.append(next_contact)
    return path
