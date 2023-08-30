## In Progress






def plot_modes():
    from scipy.interpolate import make_interp_spline


    colors = ['red','#02a8f8','#00b730','#7400ff','#434343','magenta','#fad300']
    x = np.array(tempsC)

    for pc in range(1,8):
    
        y = pcs[:,pc-1]

        X_Y_Spline = make_interp_spline(x, y)

        # Returns evenly spaced numbers
        # over a specified interval.
        X_ = np.linspace(x.min(), x.max(), 300)
        Y_ = X_Y_Spline(X_)
        
        # Plotting the Graph
        plt.ylim((-6,4))
        plt.plot(X_, -1*Y_,color=colors[pc-1])
    plt.title(f'PC Projection vs Temperature')
    plt.xlabel("Temperature °C", fontsize=12)
    plt.ylabel("Projection", fontsize=12)
    #plt.vlines(x=373,ymin=0,ymax=.84,linestyles='dotted')
    plt.legend([f'PC{i}' for i in range(1,8)], fontsize=12, loc='lower center', ncol=2)
    plt.savefig('test.pdf')


def plot_explained_variance():
    greys = ['grey' for i in range(18-7)]
    fig, ax = plt.subplots()
    colors = ['red','#02a8f8','#00b730','#7400ff','#434343','magenta','#fad300']#,'yellow']
    colors.extend(greys)

    ax.bar([str(i+1) for i in range(18)],cpca.pca.explained_variance_ratio_[:18], color=colors)
    fig.savefig('test.pdf')

def plot_loading_score_decay():
    pc = 1
    scores = cpca.sorted_norm_loadings(pc)[[f'PC{pc}']].sort_values(by='PC1',ascending=False).values
    plt.plot(scores)
    plt.hlines(.5,xmin=0,xmax=len(scores))
    plt.savefig('test.pdf')

def plot_difference_of_roots():
    # difference of roots
    p_val = np.sum(np.abs(np.diff(variance, axis=1, prepend=0)) > \
                np.abs(np.diff(original_variance, prepend=0)), axis=0) / N_permutations

    #fig, ax = plt.subplots()
    #fig.figsize((10,8))
    #plt.figure(figsize=(20,8))
    plt.scatter([f'{i+1}' for i in range(18)], p_val[:18], label='p-value on significance')
    plt.hlines(.05,xmin=0,xmax=18)
    plt.savefig('test.pdf')

def plot_contact_heatmap(heatmap):
