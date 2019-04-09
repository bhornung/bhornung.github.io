import numpy as np
from scipy.spatial import distance_matrix


class ImageVoronoiClusterBase:
    """
    Clusters image colours based on Voronoi triangulation of the 
    colour space spanned by the image.
    """
    
    @property
    def n_clusters(self):
        return self._n_clusters
    
    
    def __init__(self, n_clusters = 8):
        
        self._n_clusters = n_clusters
        self.labels_ = None
        self.score_ = None
        self.bic_ = None
    
    
    def fit(self, X):
               
        raise NotImplementedError("Fit method is not implemented")
    
    
    def fit_predict(self, X):
        
        raise NotImplementedError("fit_predict method is not implemented")
        
    
class ImageVoronoiCluster(ImageVoronoiClusterBase):
    
    def __init__(self, n_clusters = 5):
        
        super().__init__(n_clusters = n_clusters)
        
        
    def fit(self, X, weights):
        """
        Parameters:
            X (np.ndarray[n_colours, 3]) : colours
            weights (np.ndarray[n_colours]) : frequency of each colour
        """
          
        # pariwise distance to centres
        dmat = distance_matrix(X[self._n_clusters:], X[:self._n_clusters])
        
        # find closest
        labels = np.argmin(dmat, axis = 1)
        
        # collect labels
        self.labels_ = np.append(np.arange(self._n_clusters), labels)
        
        # calculate score
        self.score_ = np.dot(dmat[np.arange(labels.size),labels], weights[self._n_clusters:])
        
        return self
    
    
    def fit_predict(self, X, weights):
        
        return self.fit(X, weights).labels_
        
        
class WeightedImageVornoiCluster(ImageVoronoiClusterBase):

        def __init__(self, n_clusters = 5):
        
            super().__init__(n_clusters = n_clusters)
            
            
        def fit(self, X, weights):
            """
            Parameters:
                X (np.ndarray[n_colours, 3]) : colours
                weights (np.ndarray[n_colours]) : frequency of each colour
            """
          
            # pariwise distance to centres
            dmat = distance_matrix(X[self._n_clusters:], X[:self._n_clusters])
            
            # find closest
            labels = np.argmin(dmat, axis = 1)
            
            # collect labels
            self.labels_ = np.append(np.arange(self._n_clusters), labels)
            
            
            # calculate score
            weights_i = np.zeros(labels.shape, dtype = np.float)
            for idx in np.arange(self._n_clusters):
                keep_mask = labels == idx
                weights_i[keep_mask] = weights[idx] / keep_mask.sum()
                
            self.score_ = np.dot(dmat[np.arange(labels.size),labels], weights[self._n_clusters:] * weights_i) / self._n_clusters
            
            return self
            
        
        def fit_predict(self, X, weights):
        
            return self.fit(X, weights).labels_