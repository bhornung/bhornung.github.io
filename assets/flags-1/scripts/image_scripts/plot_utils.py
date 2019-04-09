from imageio import imread
import numpy as np
import matplotlib.pyplot as plt

from histo_utils import calculate_colour_histogram
from image_utils import ImageEncoder, apply_colour_clustering_to_flag
    
    
def plot_colour_histogram(colours, counts, ax, n_show = None):
    """
    Plots bar histogram of colour histogram.
    Parameters:
        colors (np.ndarray[n_colours, 3]) : list of colours
        counts (np.ndarray[n_colours] : counts of colours
        ax (plt.axis) : axis to draw on
        n_show ({int, None}) : how many colours to draw. If None, all colours are plotted. Default: None.
    """
    
    
    if n_show is None:
        i_max = counts.size
    else:
        i_max = min(counts.size, n_show)
    
    xvals = np.arange(i_max)
    yvals = counts[:i_max]
    
    colours_ = colours / 256.0
     
    ax.set_yscale('log')
    ax.set_facecolor('#f0f0f0')
    ax.set_xticks([])
    ax.set_xlabel('Colours')
    ax.set_ylabel('P(colour)')
    ax.bar(xvals, yvals, color = colours_)
    ax.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
    
    
def plot_flag_clustered(encoded, cluster_colour_map, ax):
    
    reconstructed = apply_colour_clustering_to_flag(encoded, cluster_colour_map)
    ax.imshow(ImageEncoder.decode(reconstructed))
    ax.axis('off')


def plot_flag_clustered_with_clusters(encoded, cluster_results, n_show = 9):
    
    histo  = calculate_colour_histogram(encoded)
    
    # set up figure
    fig = plt.figure()#gridspec_kw = {'wspace' : 0.01, 'hspace' : 0.01})
    
    fig.set_size_inches(10, 10)
    
    n_show_ = min(len(cluster_results.n_clusters), n_show)
    
    n_rows = n_show_ // 3
    if (n_show_ * 2) % 3 != 0:
        n_rows += 1
    n_rows *= 2
    
    gs = plt.GridSpec(n_rows, 3, hspace = 0.01, wspace = 0.01)
    
    # plot colour clusters
    for idx, (n_clusters, labels) in enumerate(zip(cluster_results.n_clusters, 
                                                   cluster_results.labels[:n_show_])):
        
        # plot clustered lofasz
        r = (idx // 3) * 2 + 1
        c = idx % 3
        ax = fig.add_subplot(gs[r,c] , projection='3d')

        plot_colour_cluster_results(histo.colours, 
                                     histo.counts, 
                                     labels, 
                                     np.arange(n_clusters).tolist(), 
                                     ax)
        
    # plot flags
    for idx, cm in enumerate(cluster_results.colour_maps[:n_show_]):
     
        r = (idx // 3) * 2
        c = idx % 3
        ax = fig.add_subplot(gs[r, c])
        plot_flag_clustered(encoded, cm, ax)
        

def plot_colour_cluster_results(colours, weights, labels, centres, ax, n_colours = 100):
    
    colours_ = colours / 256 
    colours_ = colours_[:min(n_colours, colours_.shape[0])]
    
    for idx, (c, w, l) in enumerate(zip(colours_, weights, labels)):
        
        if not idx in centres:
            
            shader_colour = colours_[centres[l]]
            ax.scatter(c[0], c[1], c[2], s = 100,  
                       facecolors = 'none', edgecolors = shader_colour.reshape(-1,3),
                       linewidths = 0.5)
        
            ax.scatter(c[0], c[1], c[2], c = c.reshape(-1,3), s = 100 * np.power(w, 1/3))
            
        else:
            ax.scatter(c[0], c[1], c[2], c = c.reshape(-1,3), s = 2 * 100 * np.power(w, 1/3), marker = '+')
        
        ax.set_xlim((0, 1.02))
        ax.set_ylim((0, 1.02))
        ax.set_zlim((0, 1.02))
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        
def plot_cluster_scores(n_clusters, scores, elbow_position, ax, 
                        color = 'blue', elbow_color = 'red'):
    
    ax.grid(True)
    
    for idx, (x, y) in enumerate(zip(n_clusters, scores)):
    
        color_ = [color, elbow_color][idx == elbow_position]
        marker = ['o', 'x'][idx == elbow_position]
        ax.scatter(x, y, color = color_, marker = marker)
        
    ax.plot(n_clusters, scores, linestyle  = '--', color = color)
    
    ax.set_xlim((0,16))
    ax.set_ylim(bottom = 0)
    
    ax.set_xlabel(r'$N_{C}$')
    ax.set_ylabel('score')
    
    
def plot_flag_histo_label_groups(groups_):

    n_rows = (len(groups_) // 3) * 2
    n_rows += [2, 0][len(groups_) % 3 == 0]
    
    fig, axes = plt.subplots(n_rows, 3, gridspec_kw = {'wspace' : 0.33, 'hspace' : 0.05})
    fig.set_size_inches(10, 12)
    
    for idx, (image, histo, label) in enumerate(groups_):
          
        r, c = (idx // 3) * 2, idx % 3
 
        axes[r,c].imshow(image)
        axes[r,c].axis('off')
        
        r += 1
        plot_colour_histogram(histo.colours, histo.counts, axes[r,c], n_show = 50)
        
    plt.show()