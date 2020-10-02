import networkx as nx
from networkx.exception import NetworkXNoCycle
import numpy as np
import matplotlib.pyplot as ply


DEFAULT_NODE_STYLE = {
    "node_color": "#ffffff",
    "edgecolors": "#000000",
    "node_size": 1000,
    "linewidths": 2
}


DEFAULT_EDGE_STYLE = {
    "edge_color": "grey",
    "width": 3
}


def plot_single_border(g, ax, colour_border=False):
    
    h = g.copy()
    h.remove_node(-1)

    pos = _get_linear_node_positions(h)
    xmin = min(x[0] for x in pos.values())
    xmax = max(x[0] for x in pos.values())
    pos_border = {-1: (0.5 * (xmin + xmax), 1)}
    pos.update(pos_border)
    
    _plot_graph(g, pos, ax, colour_border=colour_border)  
    
    
def plot_triangle(g, ax):
    
    pos = {
        0: (0.5, 0.5),
        1: (1.5, 0.5),
        2: (1.0, 1.0)
    }
    _plot_graph(g, pos, ax)
    
def plot_quad(g, ax, colour_border=False):
    
    pos = {
        -1: (1.0, 1.0), 
        0:  (0.5, 0.5),
        2:  (1.5, 0.5),
        1:  (1.0, 0.0)
    }
    
    _plot_graph(g, pos, ax, colour_border=colour_border)
    
def _get_linear_node_positions(g, y=0.5):
    
    try:
        nx.find_cycle(g)
        # cannot draw a line
        return

    except NetworkXNoCycle:
        # pass -- we need these cases
        pass
        
    # is the graph is a simple path?
    try:
        start, end = map(lambda x: x[0], filter(lambda x: x[1] == 1, g.degree()))
    except:
        return
    
    # find shortest path
    path = nx.shortest_path(g, start, end)

    # get positions
    pos = {x : ((i + 0.5), y) for i, x in enumerate(path)}

    return pos
    

def plot_path_graph(g, ax):
    
    pos = _get_linear_node_positions(g)
    _plot_graph(g, pos, ax)
    
    
def _plot_graph(g, pos, ax, colour_border=False):
    
    node_style = dict(DEFAULT_NODE_STYLE.items())
    
    if colour_border:
        node_color = [
            [node_style["node_color"], "#add8e6"][k < 0] 
            for k in g.nodes
        ]
            
        node_style.update({"node_color": node_color})
    
    nx.draw_networkx_nodes(g, pos, ax=ax, **node_style)
    nx.draw_networkx_edges(g, pos, ax=ax, **DEFAULT_EDGE_STYLE)
    
    xmin = min(x[0] for x in pos.values()) - 0.5
    xmax = max(x[0] for x in pos.values()) + 0.5
    ymin = min(x[1] for x in pos.values()) - 0.5
    ymax = max(x[1] for x in pos.values()) + 0.5
    
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    ax.axis("off")

