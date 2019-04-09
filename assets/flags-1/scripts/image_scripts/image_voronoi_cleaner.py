import sys

from cluster_utils import BatchClusterer
from histo_utils import calculate_colour_histogram
from image_utils import ImageEncoder
from image_utils import apply_colour_clustering_to_flag

from image_voronoi import ImageVoronoiCluster


class ImageVoronoiCleaner:
    
    @property
    def histo(self):
        return self._histo
    
    
    def __init__(self):
        
        self._histo = None
        
        
    def clean(self, X):
        """
        Replaces the satellite colours with main colours in an image.
        Parameters:
            X (np.ndarray[height, width, {3:4}]) : R,G,B[,A] image
        
        Returns:
            cleaned (np.ndarray[height, width, 3}]) : cleaned R,G,B image
        """
        
        # convert to RRGGBB
        encoded = ImageEncoder.encode(X)
        
        # perform series of clustering
        bcl = BatchClusterer(ImageVoronoiCluster, max_clusters = 15)
        bcr = bcl.process(encoded)
        
        # select best colour map
        cluster_colour_map = bcr.colour_maps[bcr.elbow_position]
        
        # replace satelite colours with main colours
        applied = apply_colour_clustering_to_flag(encoded, cluster_colour_map)
        
        # transform to (RR,GG,BB)
        cleaned = ImageEncoder.decode(applied)
        
        
        # create histogram
        self._histo = calculate_colour_histogram(applied)
        
        return cleaned