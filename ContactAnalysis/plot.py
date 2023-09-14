import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
from ChACRA.ContactAnalysis.utils import chacra_colors

# example darkmode/neon 
# https://towardsdatascience.com/cyberpunk-style-with-matplotlib-f47404c9d4c5
'''
plt.style.use("dark_background")
for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # bluish dark grey
colors = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41',  # matrix green
]

fig, ax = plt.subplots()
df['Potential Energy (kJ/mole)'].plot(marker='o', color=colors, ax=ax)
# Redraw the data with low alpha and slighty increased linewidth:
n_shades = 10
diff_linewidth = 1.05
alpha_value = 0.3 / n_shades
for n in range(1, n_shades+1):
    df['Potential Energy (kJ/mole)'].plot(marker='o',
            linewidth=2+(diff_linewidth*n),
            alpha=alpha_value,
            legend=False,
            ax=ax,
            color=colors)
# Color the areas below the lines:

#for column, color in zip(df, colors):
#    ax.fill_between(x=df.index,
#                    y1=df[column].values,
#                    y2=[0] * len(df),
#                    color=color,
#                    alpha=0.1)
                    
ax.grid(color='#2A3459')
#ax.set_xlim([ax.get_xlim()[0] - 0.2, ax.get_xlim()[1] + 0.2])  # to not have the markers cut off
#ax.set_ylim(0)
plt.show()
'''


cherenkov_blue = '#00bfff'
def plot_difference_of_roots(cpca,n_pcs=None,filename=None):
    '''
    Plot the difference of roots test results
    Parameters
    ----------
    cpca : ChACRA.ContactAnalysis.ContactFrequencies.ContactPCA object
            with permutated_explained_variance method previously called
    n_pcs : 
    '''
    if cpca._permutated_explained_variance is None:
        print("You have to run the ContactPCA.permutated_explained_variance \n method before "
              "you can plot the difference of roots results. ")
    else:
        fig, ax = plt.subplots()
        variance = cpca._permutated_explained_variance
        original_variance = cpca.pca.explained_variance_ratio_
        if n_pcs == None:
            n_pcs = variance.shape[1]
        # difference of roots
        p_val = np.sum(np.abs(np.diff(variance, axis=1, prepend=0)) > \
                    np.abs(np.diff(original_variance, prepend=0)), axis=0) / cpca._N_permutations
        ax.hlines(.05,xmin=0,xmax=n_pcs,color='r',zorder=1)
        ax.scatter([f'{i+1}' for i in range(n_pcs)], p_val[:n_pcs], color=cherenkov_blue, label='p-value on significance')
        if filename is not None:
            fig.savefig(filename)
    
def plot_chacras(cpca, n_pcs=4, contacts=None, temps=None, filename=None):
    '''
    Plot the projections of n principal components
    smoothing is applied
    #TODO find that pretty plotting library and give the lines a plasma glow
    '''
    if temps is None and contacts is not None:
        print('Using the axis labels from the contact data as temperature labels. '
              'If this is incorrect, you can supply the list of temperatures.')
        temps = list(cont.freqs.index)
    if temps is not None:
        if len(temps) != cpca.pca.components_.shape[0]:
            print("The temperature (or x axis) list does not contain the same number of entries "
                  "as there are rows in the principal components.")
    # temp_units = °C

    fig, ax = plt.subplots()
    x = np.array(temps)
    pcs = cpca._transform
    for pc in range(1,n_pcs+1):
        y = pcs[:,pc-1]

        X_Y_Spline = make_interp_spline(x, y)

        # Returns evenly spaced numbers
        # over a specified interval.
        X_ = np.linspace(x.min(), x.max(), 300)
        Y_ = X_Y_Spline(X_)
        
        # Plotting the Graph
        #plt.ylim((-6,4))
        ax.plot(X_, -1*Y_,color=chacra_colors[pc-1])
                
    ax.set_title(f'ChACRA Modes')
    ax.set_xlabel("Temperature ", fontsize=12)
    ax.set_ylabel("Projection", fontsize=12)
    #ax.vlines(x=373,ymin=0,ymax=.84,linestyles='dotted')
    ax.legend([f'PC{i}' for i in range(1,n_pcs+1)], fontsize=12, loc='lower center', ncol=2)
    if filename:
        ax.savefig(filename)
    
    

