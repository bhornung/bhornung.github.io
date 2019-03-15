"""
Misc helper utils
"""
from collections import Counter
import numpy as np


def calc_keyed_prob_dist(collection):
    """
    Calculates the keyed probability distribution of a collection.
    """

    counter = Counter(collection)
    _ = counter.pop(-1, None)
    n_all_counts = sum(counter.values())

    prob_dict = dict((k, v / n_all_counts) for k, v in counter.items())
    
    return prob_dict


def calc_set_prob_sim(prob_dict1, prob_dict2):
    """
    Parameters:
        prob_dict1 ({object: float}) : keyed probability distribution
    """
    
    all_keys = set(prob_dict1.keys()) | set(prob_dict2.keys())
    n_keys = len(all_keys)
    sum_ = sum(abs(prob_dict1.get(k, 0) - prob_dict2.get(k, 0)) for k in all_keys)
    
    sim_ = 1.0 - sum_ / n_keys
    
    return sim_


def calculate_pairwise_set_prob_sim(collection):
    """
    Parameters:
        prob_dict ([{object: float}]) : list of keyed probability distributions.
    """

    collector = []
    
    for i, coll1 in enumerate(collection):
        for j, coll2 in enumerate(collection[i:]):
            
            sim_ = calc_set_prob_sim(coll1, coll2)
            if sim_ != 0:
                collector.append((i, j + 1, sim_))
    
    result = np.array(collector).T
    
    return result
    
    
def calculate_backward_number_ratio(counters):
    """
    Ratio of number of elements from previous sets for each set. 
    """
    
    ratios = np.zeros(len(counters))
    
    cumset = set()
    
    for idx, counter in enumerate(counters):
        ratios[idx] = sum(counter.get(k, 0) for k in cumset) / sum(counter.values())
        cumset |= set(counter.keys())
        
    return ratios


def calculate_forward_number_ratio(counters):
    """
    Ratio of number of elements from a given set for all succeeding sets.
    """
    
    n_sets = len(counters)
    ratios = np.zeros(n_sets)
    
    cumset = set()
    cumcount = 0
    
    for idx, counter in enumerate(counters[::-1]):
        try:
            ratios[n_sets - 1 - idx] = sum(counter.get(k, 0) for k in cumset) / cumcount
        except:
            pass
        
        cumset |= set(counter.keys())
        cumcount += sum(counter.values())
        
    return ratios


def calculate_backward_resolved_number_ratio(counters):
    """
    Ratio of number of elements from previous sets summed over previous sets.
    """
    
    collector = []
    
    cntr_keys = [set(x.keys()) for x in counters]
    cntr_sums = [sum(x.values()) for x in counters]
    
    for i, (cntr, cntr_sum) in enumerate(zip(counters, cntr_sums)):
        for j, keys in enumerate(cntr_keys[:i]):
        
            ratio = sum((cntr.get(k, 0) for k in keys)) / cntr_sum
            
            if ratio > 0:
                collector.append((i, j, ratio))

    result = np.array(collector).T 
    
    return result
    

def calculate_weighted_jaccard(counters, with_diag = False):
    
    collector = []
    
    for i, cntr1 in enumerate(counters):
        for j, cntr2 in enumerate(counters[:i]):
            numerator = sum(x for x in (cntr1&cntr2).values())
            denominator = sum(x for x in (cntr1 | cntr2).values())
            sim_ = numerator / denominator
            
            if sim_ != 0:
                collector.append((i, j, sim_))
    if with_diag:
        collector.extend([(i,i,1.0) for i in range(len(counters))])
            
    return np.array(collector).T