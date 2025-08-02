import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from .visualize.colors import *
from itertools import combinations

def plot_difference_of_roots(cpca,
                             n_pcs=None,
                             filename=None, 
                             dot_color=cherenkov_blue, 
                             cutoff_color='r'):
    '''
    Plot the difference of roots test results
    Parameters
    ----------
    cpca : ChACRA.ContactAnalysis.ContactFrequencies.ContactPCA object
            with permutated_explained_variance method previously called
    n_pcs : int 
        The number of pcs to include in the plots
    dot_color : str
        Specify the color of dots in the plot.
    cutoff_color : str  
        Specify the color of the horizontal bar in the plot.
    filename : str
        If saving the file, provide the filename.
    '''
   
    if cpca._permutated_explained_variance is None:
        print("You have to run the ContactPCA.permutated_pca \n method before "
              "you can plot the difference of roots results. ")
    else:
        fig, ax = plt.subplots()
        variance = cpca._permutated_explained_variance
        original_variance = cpca.pca.explained_variance_ratio_
        if n_pcs == None:
            n_pcs = variance.shape[1]
        # difference of roots
        # can just do cpca.chacra_pvals 
        p_val = np.sum(np.abs(np.diff(variance, axis=1, prepend=0)) > \
                    np.abs(np.diff(original_variance, prepend=0)), axis=0) / \
                        cpca._N_permutations
        ax.hlines(.05,xmin=0,xmax=n_pcs,color=cutoff_color,zorder=1)
        ax.scatter([f'{i+1}' for i in range(n_pcs)], p_val[:n_pcs], 
                   color=dot_color, label='p-value on significance')
        if filename is not None:
            fig.savefig(filename)
    
def plot_chacras(cpca, 
                 n_pcs=4, 
                 contacts=None, 
                 temps=None, 
                 colors=chacra_colors,
                 spacing='geometric', 
                 temp_scale=None,
                 filename=None):
    '''
    cpca : ContactPCA
        The ContactPCA object.
    n_pcs : int
        The number of PCs to include in the plot.
    contacts : ContactFrequencies (remove)
        provide the contact frequencies object to get the temps from the index.
    temps : list
        List of temperatures.
    colors : list
        List of hex colors for the lines.
    
    spacing : string
        'geometric' or 'linear'.
        Specifies how the temperature points are distributed. 

    Plot the projections of n principal components
    smoothing is applied
    #TODO offer plasma glow 
    '''
    if temps is None and contacts is not None:
        print('Using the axis labels from the contact data as temperature labels. '
              'If this is incorrect, you can supply the list of temperatures.')
        try:
            temps = list(contacts.freqs.index)
        except:
            temps = list(contacts.index)
    elif temps is not None:
        if len(temps) != cpca.pca.components_.shape[0]:
            print("The temperature (or x axis) list does not contain the same number of entries "
                  "as there are principal components.")
    elif temps is None:
        temps = list(range(cpca.loadings.shape[1]))
        print("No temperatures or contact frequency data provided to obtain temperatures from. X-axis corresponds to system id.")

    fig, ax = plt.subplots()
    x = np.array(temps)
    pcs = cpca._transform
    for pc in range(1,n_pcs+1):
        y = pcs[:,pc-1]
        # smooth the data
        X_Y_Spline = make_interp_spline(x, y)

        # Returns evenly spaced numbers
        # over a specified interval.
        if spacing == 'linear':
            X_ = np.linspace(x.min(), x.max(), 300)
        elif spacing == 'geometric':
            X_ = np.geomspace(x.min(), x.max(), 300)
        Y_ = X_Y_Spline(X_)
        
        # Plotting the Graph
        #plt.ylim((-6,4))
        ax.plot(X_, 1*Y_,color=colors[pc-1])
                
    ax.set_title(f'ChACRA Modes')
    if temp_scale is not None:
        ax.set_xlabel(f"Temperature ({temp_scale})", fontsize=12)
    else:
        ax.set_xlabel(f"Temperature ({temp_scale})", fontsize=12)
    ax.set_ylabel("Projection", fontsize=12)
    #ax.vlines(x=373,ymin=0,ymax=.84,linestyles='dotted')
    ax.legend([f'PC{i}' for i in range(1,n_pcs+1)], fontsize=12, 
              loc='lower center', ncol=2)
    if filename:
        fig.savefig(filename)

def biplots(cpca, 
            pcs=list(range(1,5)), 
            label_top=None, 
            colors=chacra_colors, 
            filename=None):
    '''
    NOT IMPLEMENTED 
    
    option to label outliers/ corners/ top loading scores and color on a mixing 
    gradient so something that's in a corner of a pc1-pc2 biplot appears purple

    quick way to identify or look for correlation with known functionally 
    important residues and suggest where connections between modes occur.
    
    Parameters
    ----------

    cpca : ChACRA.ContactAnalysis.ContactPCA

    label_top : int
        number of top scoring contacts to display labels for on the plot
    '''
    # TODO when you add labels, the figure size changes 
    # (probably because of "constrained_layout")
    # if pcs is list of int, make all the combos
    if type(pcs[0]) == int:
        combos = list(combinations(pcs,2))

    # if it's a list of tuples, plot just the tuple combos
    elif type(pcs[0]) == tuple:
        combos = pcs

    n_plots = len(combos)
    # plot in rows of 4
    n_rows = int(np.ceil(n_plots/4))
    # remove the last n_plot ax
    n_axes_to_remove = 4 - int(n_plots % 4)
    fig, axs = plt.subplots(n_rows, 4, figsize=(15,3*n_rows), 
                            constrained_layout=True)
    for i, combo in enumerate(combos):
        pc_a = combo[0]
        pc_b = combo[1]
        # get the color gradient from transparent white to the pc's hex code
        # TODO Everything goes totally transparent and colors are all red now...
        a_grad = get_color_gradient('#ffffff00',colors[pc_a-1],100, 
                                    return_rgb=True,alpha=True)
        b_grad = get_color_gradient('#ffffff00',colors[pc_b-1],100, 
                                    return_rgb=True, alpha=True)

        # indexing the list of colors by the normalized loading score 
        a_grad_len = len(a_grad)-1
        b_grad_len = len(b_grad)-1


        labels = cpca.sorted_norm_loadings(pc_a)[f'PC{pc_a}'].index
        # get the normalized values to establish the gradient
        val_1 = cpca.sorted_norm_loadings(pc_a)[f'PC{pc_a}'].values
        val_2 = cpca.sorted_norm_loadings(pc_a)[f'PC{pc_b}'].values
        # mix the two colors for all of the points
        colors = [rgb_to_hex(mix_color(
                                    a_grad[round(val[0]*a_grad_len)], 
                                    b_grad[round(val[1]*b_grad_len)]
                                        )
                            ) 
                for val in zip(val_1,val_2)
                ]
        # then use the original loadings for plotting
        val_1 = cpca.loadings[f'PC{pc_a}'].loc[labels].values
        val_2 = cpca.loadings[f'PC{pc_b}'].loc[labels].values
        # option to label the top n scoring contacts
        ax = axs.flatten()[i]
        ax.scatter(val_1,val_2,c=colors);
        ax.set_xlabel(f'PC{pc_a}')
        ax.set_ylabel(f'PC{pc_b}')
        
        if label_top is not None:
            top_labels = list(cpca.sorted_norm_loadings(pc_a).index[:label_top])
            top_labels.extend(list(cpca.sorted_norm_loadings(pc_b).index[:label_top]))

            labels = [label if label in top_labels else None for label in labels]
        
            [ax.text(h, j, labels[k], **{'fontsize':'x-small'}) 
                    for k,(h,j) in enumerate(zip(val_1, val_2))];
    
    if n_axes_to_remove > 0:
        for r in range(1,n_axes_to_remove+1):
            axs.flatten()[-r].remove()
    #fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)

def plot_loadings():
    '''
    Plot the loading score values on a pc in descending order
    '''
    pass

def plot_loading_distributions():
    '''
    Violin plots of loading scores.
    '''
    pass

'''
# plot the CombinedChacra standouts
fig, axs = plt.subplots(5,6, figsize=(14,8),sharey=True)
for i, contact in enumerate(different_stable):
    axs.flatten()[i].plot(temps, ensemblea.freqs[contact].values)
    axs.flatten()[i].plot(temps, ensembleb.freqs[contact].values)
    axs.flatten()[i].set_title(contact)
fig.tight_layout()

'''

def plot_energies(energies, filename=None, n_bins=20):
    ''''
    energies : np.array
        The n_cycles x n_states array of energies from the replica exchange 
        simulations.
    
    Returns
    -------
    matplotlib.pyplot.plt
    '''
    for i, rep in enumerate(range(energies.shape[1])):
        plt.hist(energies[:,rep],label=i, bins=n_bins)
        #plt.legend(loc='upper left')
    if filename is not None:
        plt.savefig(filename)
