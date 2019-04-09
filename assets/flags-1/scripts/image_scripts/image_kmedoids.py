import numpy as np
from scipy spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform

class KMedoidsBase:
    
    @property
    def n_init(self):
        return self._n_init
    
    
    @property
    def max_iter(self):
        return self._max_iter
    
    
    @property
    def n_clusters(self):
        return self._n_clusters
    
    
    @property
    def n_max_clusters(self):
        return self._n_max_clusters
    
    def __init__(self, n_clusters = 5, 
                 n_max_clusters = 10, 
                 n_init = 5, 
                 max_iter = 200):
        
        if n_init < 1:
            raise ValueError("'n_init' must be positive")
            
        self._n_init = n_init
        
        self._n_clusters = n_clusters
        
        self._n_max_clusters = n_max_clusters
        
        self._max_iter = max_iter
             
         # output variables
        self.cluster_centres_ = None
        self.labels_ = None
        self.score_ = np.finfo(np.float).max
        
        
    def fit(self, X, weights = None):
        
        raise NotImplementedError()
        
    def fit_predict(self, X, weights = None):
        
        raise NotImplementedError()
        
        
class KMedoids(KMedoidsBase):
    
    def __init__(self, n_clusters = 5, n_max_clusters = 10, n_init = 5, max_iter = 200):
        
        super().__init__(n_clusters = n_clusters, 
                         n_max_clusters = n_max_clusters, 
                         n_init = n_init, 
                         max_iter = max_iter)
        
        
    def fit(self, X, weights):
        """
        Performs a series of k-medoids clustering and select the one with the lowest variace.
        Parameters:
            X (np.ndarray[n_samples, n_features] of float) : sample
            w (np.ndarray[n_samples]) : sample weights
        """
        
        for idx in np.arange(self.n_init):
            
            # initialise cluster centres based of datum weight
            centre_idcs = _init_cluster_centres(weights, self.n_clusters, self.n_max_clusters)
            
            # perform one clustering
            cluster_centres, labels, score = _kmedoids_fit_one(X, weights, 
                                                               self.n_clusters, self.max_iter,
                                                               centre_idcs)
            
            # select best clustering
            if score < self.score_:
                self.cluster_centres_ = cluster_centres
                self.labels_ = labels
                self.score_ = score
                
        return self
    
    
    def fit_predict(X, weigths):
        
        return self.labels_
    
    
def _init_cluster_centres(w, n_clusters, n_max_clusters):
    """
    Select cluster centres.
    Parameters:
        w (np.ndarrray[n_samples] of float) : weigths of the points
        n_clusters (int) : number of cluster centres
        n_max_cluster_centres (int) : upper limit of number of cluster centres
    
    Returns:
        centre_idcs (np.ndarray[n_clusters] of int) : indices of the cluster centres
    """

    if w.ndim != 1:
        raise ValueError("'w' must be onedimensional. Got: {0}".format(w.shape))
    
    if n_clusters > n_max_clusters:
        raise ValueError("'n_clusters' must be smaller or equal to 'n_max_cluster_centres'. Got: {0} < {1}"
                        .format(n_clusters, n_max_clusters))
        
    if n_max_clusters > w.size:
        raise ValueError("'n_max_clusters' must be smaller or equal to size of w. Got: {0} < {1}"
                          .format(n_clusters, w.size))
    
    prob = - 1.0 / np.log(w[:n_max_clusters])
    prob = prob / prob.sum()
    
    centre_idcs = np.random.choice(np.arange(n_max_clusters), size = n_clusters, 
                                   replace = False, p = prob)
    
    np.sort(centre_idcs)
    
    return centre_idcs

    
def _kmedoids_fit_one(X, w, n_clusters, max_iter, centre_idcs):
    """
    Performs one fit of k-medoid clustering.
    Parameters:
        X (np.ndarray[n_samples, n_features] of float) : sample
        w (np.ndarray[n_samples]) : sample weights
        n_clusters (int) : number of clusters
        max_iter (int) : number of maximum EM cycles
        centre_idcs (np.ndarray[n_clusters] of int) : initial cluster centres 
    """
    
    # choose n_clusters centre
    c_idcs = centre_idcs
    
    c_idcs_new = np.zeros_like(c_idcs)
    
    dmat = squareform(pdist(X))
    
    for idx in range(max_iter):
        
        # maximisation
        i_closest = np.argmin(dmat[c_idcs, :], axis = 0)
        
        # expectation
        for ic_idx in np.arange(c_idcs.size):
            
            # select points that are closest to a cluster centre
            keep_mask = i_closest == ic_idx
             
            # calculate weigted average of the cluster
            xc_average = np.average(X[keep_mask], weights = w[keep_mask], axis = 0)
             
            # find medoid
            c_idcs_new[ic_idx] = np.argmin(distance_matrix(X, xc_average.reshape(-1, 3)))
            print(idx, ":", c_idcs_new[ic_idx], c_idcs[ic_idx])
        # cluster centres do not change
        if np.abs(c_idcs_new - c_idcs).sum() == 0:
            c_idcs = c_idcs_new + 0
            break
            
        else:
            c_idcs = c_idcs_new + 0
    
    # prepare output variables
    cluster_centres = X[c_idcs]
    
    labels = np.argmin(dmat[c_idcs, :], axis = 0)
    
    score = np.sum(np.min(dmat[c_idcs, :], axis = 0))
    
    return cluster_centres, labels, score