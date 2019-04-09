import numpy as np

from histo_utils import calculate_colour_histogram
from image_utils import create_cluster_colour_map

class BatchClusterer:
    """
    Lightweight wrapper for batch clustering over a range of number of clusters.
    """
    
    @property
    def clusterer(self):
        return self._clusterer.__str__()
    
    @property
    def max_clusters(self):
        return self._max_clusters
    
    
    def __init__(self, clusterer, max_clusters = 10, clusterer_kwargs = {}):
        """
        Parameters:
            clusterer (object) : clusterer, It should:
                * have a fit() method
                * with the signature fit(X, w)
            max_clusters (int) : maximum number of clusters
            clusterer_kwargs ({:}) : optional kwargs for clusterer.init()
        """
        
        self._clusterer = clusterer
        self._clusterer_kwargs = clusterer_kwargs
        self._max_clusters = max_clusters
        
        
    def process(self, encoded_image):
        """
        Performs a series of clusterings on an image.
        Parameters:
            image (np.ndarray([width, height])) : RRGGBB image
            
        Returns:
            results (BatchClusterResult) : summary of the clusterings.
        """
        
        histo = calculate_colour_histogram(encoded_image)
        
        # reset number of maximum clusters, if needed
        if histo.counts.size < self.max_clusters:
            self._max_clusters = histo.counts.size
            
        # intialise temp containers
        n_cluster_list, score_list, label_list, colour_map_list = [], [], [], []
        
        # perform a sequence of clsuterings
        for n_clusters in range(1, self.max_clusters + 1):

            # create a clusterer and fit histogram of colours
            clusterer = self._clusterer(n_clusters = n_clusters, **self._clusterer_kwargs)
            clusterer.fit(histo.colours, histo.counts)
            
            # save clustering dependent output variables
            n_cluster_list.append(clusterer.n_clusters)
            score_list.append(clusterer.score_)
            label_list.append(clusterer.labels_)
            cluster_colour_map = create_cluster_colour_map(histo.codes, 
                                                           np.arange(clusterer.n_clusters), 
                                                           clusterer.labels_)
            colour_map_list.append(cluster_colour_map)

        # collate results
        result = BatchClusterResult(n_cluster_list, 
                                    score_list, 
                                    label_list, 
                                    colour_map_list)

        return result
        
    
class BatchClusterResult:
    """
    Container class for collection of clustering results.
    """
    
    @property
    def n_clusters(self):
        return self._n_clusters
    
    @property
    def scores(self):
        return self._scores

    @property
    def labels(self):
        return self._labels
    
    @property
    def colour_maps(self):
        return self._colour_maps
        
    
    @property
    def elbow_position(self):
        return self._elbow_position
    
    
    def __init__(self, n_clusters, scores, labels, colour_maps):
        """
        Stores clustering results.
        Parameters:
            n_clusters ([int]) : list of number of clusters
            scores ([float]) : list of scores
            bic ([float]) : list of Bayesian information criterion scores
            labels ([np.ndarray]) : list of labels from various clusterings
            colour_maps ([{np.ndarray : np.ndarray}]) : colour map that represents the colour substitution
        """
        
        self._n_clusters = n_clusters
        self._scores = scores
        self._labels = labels
        self._colour_maps = colour_maps
        
        # calculate elbow position
        self._elbow_position = find_elbow_by_distance(self.scores)
    
    
def find_elbow_by_distance(scores):
    """
    Finds the elbow point. ~ is defined as the point on the (n_clusters, scores)-curve,
    whose distance is the maximal to the line connecting the curves first and last points.
    Parameters:
        scores ([int]) : list of scores
        
    Returns:
        elbow_position (int) : index of the elbow point
    """
    
    n_points = len(scores)
    
    if n_points == 0:
        raise ValueError("Empty scores list")
    
    elif n_points == 1:
        return 0
    
    elif n_points == 2:
        return [0,1][scores[0] < scores[1]]
    
    
    secant = np.array([[0, scores[0]],
                      [n_points, scores[-1]]]
                     )
    
    score_vec = np.array([np.arange(0, n_points), 
                         np.array(scores)]
                        )
    
    delta_x, delta_y = secant[1] - secant[0]
    det = secant[0,1] * secant[1,0] - secant[0,0] * secant[1,1]

    distances = np.abs(delta_y * score_vec[0] - delta_x * score_vec[1] + det)
    elbow_position = np.argmax(distances)
    
    return elbow_position
