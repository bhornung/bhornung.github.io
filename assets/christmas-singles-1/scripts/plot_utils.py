import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator


def create_connecting_arc(p1, p2, color = 'blue', lw = 1, top = True):
    
    width = abs(p1[0] - p2[0])
    height = width / 30
    centre = (
                (p1[0] + p2[0]) / 2, 
                (p1[1] + p2[1]) / 2
             ) 
    
    if top:
        theta1, theta2 = 0, 180
    else:
        theta1, theta2 = 180, 0
        
    arc = Arc(centre, width, height, 
              theta1 = theta1, theta2 = theta2,
              lw = lw, edgecolor = color)
    
    return arc
    
def plot_linear_graph(E, ax):
    """
    
    """
    
    # shift nodes
    n_min = E[:2].min()
    n_max = E[:2].max()
    n_nodes =  n_max - n_min + 1
    
    w_max = E[2].max()
    
    # node coordinates
    x = np.arange(n_nodes)
    y = np.full_like(x, 0)
    
    # loop over edges
    for edge in E.T:
        if edge[2] == 0:
            continue
            
        lw = edge[2] / w_max * 5
        
        arc = create_connecting_arc((edge[0] - n_min, 0), (edge[1] - n_min, 0),
                                    lw = lw, top = True, color = 'navy')
        ax.add_patch(arc)

    # add cicrles
    ax.scatter(x, y, marker = 'o', facecolors ='none', edgecolors = 'black')
    ax.set_ylim((-0.1, 1.0))
    ax.axis('off')
    

def plot_linear_graph_with_colors(E, ax, colors):
    """
    
    """
    
    # shift nodes
    n_min = E[:2].min()
    n_max = E[:2].max()
    n_nodes =  n_max - n_min + 1
    
    w_max = E[2].max()
    
    # node coordinates
    x = np.arange(n_nodes)
    y = np.full_like(x, 0)
    
    # loop over edges
    for edge in E.T:
        if edge[2] == 0:
            continue
        if edge[0] == edge[1]:
            continue
            
        if colors[int(edge[0])] != colors[int(edge[1])]:
            continue
            
        arc = create_connecting_arc((edge[0] - n_min, 0), (edge[1] - n_min, 0),
                                    lw = 2, top = True, color = colors[int(edge[0])])
        ax.add_patch(arc)

    # add cicrles
    ax.scatter(x, y, marker = 'o', c = colors)
    ax.set_ylim((-0.1, 1.0))
    ax.axis('off')


def plot_scatter_histo(X, ax, cmap = plt.get_cmap('winter'), 
                       scatter_kwargs = {}, plot_kwargs = {}):
    """
    Plots a scatter histogram
    """
    
    scatter_size = scatter_kwargs.get('size', 10)
    scatter_marker = scatter_kwargs.get('marker', 'o')
    label = plot_kwargs.get('label', None)
    
    
    cs = ax.scatter(X[0], X[1], c = cmap(X[2]),
              norm = LogNorm(vmin = X[2].min(), vmax = X[2].max()),
              marker = scatter_marker, s = scatter_size,
              label = label)
              
    ax.legend()
    
    x_min = plot_kwargs.get('xmin', 0)
    x_max = plot_kwargs.get('xmax', X[0].max())
    y_min = plot_kwargs.get('ymin', 0)
    y_max = plot_kwargs.get('ymax', X[1].max())
    
    if 'xticks' in plot_kwargs:
        ax.set_xticks(plot_kwargs['xticks'])
        
    if 'xticklabels' in plot_kwargs:
        ax.set_xticklabels(plot_kwargs['xticklabels'])
        
    if 'yticks' in plot_kwargs:
        ax.set_yticks(plot_kwargs['yticks'])
        
    if 'yticklabels' in plot_kwargs:
        ax.set_yticklabels(plot_kwargs['yticklabels'])
    
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_aspect(1)
    
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha = 0.5)
    ax.grid(b=True, which='minor', color='grey', linestyle='--', alpha = 0.25)
    
    ax.set_facecolor('whitesmoke')
    
    if 'major_ticks' in plot_kwargs:
        ax.xaxis.set_major_locator(MultipleLocator(plot_kwargs['major_ticks']))
        ax.yaxis.set_major_locator(MultipleLocator(plot_kwargs['major_ticks']))
    
    if 'minor_ticks' in plot_kwargs:
        ax.xaxis.set_minor_locator(MultipleLocator(plot_kwargs['minor_ticks']))
        ax.yaxis.set_minor_locator(MultipleLocator(plot_kwargs['minor_ticks']))
    
    xlabel = plot_kwargs.get('xlabel', '')
    ylabel = plot_kwargs.get('ylabel', '')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if 'xticks' in plot_kwargs:
        ax.set_xticks(plot_kwargs['xticks'])

    if 'yticks' in plot_kwargs:
        ax.set_yticks(plot_kwargs['yticks'])
                      
    if 'xticklabels' in plot_kwargs:
        ax.set_xticklabels(plot_kwargs['xticklabels'])
        
    if 'yticklabels' in plot_kwargs:
        ax.set_xticklabels(plot_kwargs['yticklabels'])
        
    return cs
    
    
def plot_histo(X, ax, color, bins = 100, plot_kwargs = {}):
    
    label = plot_kwargs.get('label', None)
    
    ax.hist(X, bins = bins, density = False, alpha = 0.5, color = color, label = label)
    #ax.hist(X, bins = bins, cumulative = 1, density = True, histtype = 'step', lw = 2, color = color)
    
    if 'logx' in plot_kwargs:
        ax.set_xscale("log")
    
    if 'logy' in plot_kwargs:
        ax.set_yscale('log')
    
    if ('xmin' in plot_kwargs) and ('xmax' in plot_kwargs):
        ax.set_xlim((plot_kwargs['xmin'], plot_kwargs['xmax']))
    
    if ('ymin' in plot_kwargs) and ('ymax' in plot_kwargs):
        ax.set_xlim((plot_kwargs['ymin'], plot_kwargs['ymax']))
        
    ax.set_xlabel(plot_kwargs.get('xlabel', ""))
    ax.set_ylabel(plot_kwargs.get('ylabel', ""))
    ax.legend()
    
    ax.grid(True)