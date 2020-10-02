from itertools import chain
import numpy as np
import matplotlib.pyplot as plt

def _sort_level(g, level):

    degrees = [0]  *len(level)
    for i, p1 in enumerate(level):
        for j, p2 in enumerate(level[i:]):
            if ((p1, p2) in g.edges) or ((p2, p1) in g.edges):
                degrees[i] += 1
                degrees[j+i] += 1
         
    idcs = np.argsort(np.array(degrees))
    level_ = np.array(level)[idcs]

    level_ = np.hstack([level_[0:-1][::2], level_[-1], level_[1:-1][::2][::-1]])

    return level_.tolist()

def _get_levels_single(g):
    
    all_parents = set([-1])
    parents = list(g.neighbors(-1))
    parents = _sort_level(g, parents)
    
    levels = [[-1], parents]
    
    while True:
        
        all_parents |= set(parents)
        children = []
        for p in parents:
            chld = g.neighbors(p)

            # keep some order
            chld = list(filter(lambda x: x not in children, chld))
            chld = list(filter(lambda x: x not in all_parents, chld))
            children += chld
    
        if len(children) == 0:
            break
            
        parents = children
        levels.append(parents)
        
    return levels


def _get_positions_single(levels):
    
    ys = np.hstack([np.repeat(-i, len(l)) for i, l in enumerate(levels)])
    xs = np.hstack([np.arange(len(l)) - len(l)/ 2 for l in levels])
    pos = {i: (x, y) for i, x, y in zip(chain.from_iterable(levels), xs, ys)}
    
    return pos


def _draw_nodes(pos, ax):
    
    x, y = zip(*pos.values())
    colours = ["#aaeeff"] + ["white"] * (len(x) -1)
    ax.scatter(x, y, marker="o", s=75,
        facecolors=colours, edgecolors="black"
    )
    

def _draw_parent_children(g, parents, children, pos, ax):
    
    for p in parents:
        for c in children:
            if (p, c) not in g.edges():
                continue
                
            x = [pos[p][0], pos[c][0]]
            y = [pos[p][1], pos[c][1]]
            
            ax.plot(x, y, color="grey", lw=1, ls="-", zorder=-1)


def _create_curve(pos1, pos2, n_points=200, b=0.5):
    
    xb, xe = sorted([pos2[0], pos1[0]])
    a = (xe - xb) / 2

    x = np.linspace(-a, a, num=100)
    y = - b / a * np.sqrt(a * a - x * x) + pos1[1]
    x = x + a +  xb
    
    return x, y


def _draw_parent_parent(g, parents, pos, ax):
    
    for i, p1 in enumerate(parents):
        for p2 in parents[i + 1:]:
            
            if (p1, p2) not in g.edges:
                continue
                
            x, y = _create_curve(pos[p1], pos[p2])
            ax.plot(x, y, zorder=-1, lw=1, color="grey")
            


def plot_level_graph_single(g, ax):
    
    levels = _get_levels_single(g)
    pos = _get_positions_single(levels)

    for l1, l2 in zip(levels[:-1], levels[1:]):
        _draw_parent_children(g, l1, l2, pos, ax)
        
    _draw_nodes(pos, ax)
    
    for l in levels:
        _draw_parent_parent(g, l, pos, ax)
        
    ax.axis("off")
    
    xmax = max(len(l) for l in levels)
    
    ax.set_xlim((-xmax / 2 - 1, xmax / 2 + 1))
    ax.set_ylim(( - len(levels) - 0.5, 1))

