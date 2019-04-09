from collections import Counter

import numpy as np
from scipy.spatial import distance_matrix

from histo_utils import calculate_colour_histogram
from image_utils import ColourEncoder, ImageEncoder
from image_utils import apply_colour_clustering_to_flag


def _calculate_counts(X):
    """
    Calculates the counts of unique elements in an array.
    Parameters:
        X (np.ndarray of hashable)
        
    Returns:
        count_dict (Counter // {hashable : int}) : counts of individual elements 
    """

    counter = Counter(X.flat)
    
    return counter


def _shrunk(X, ratio, rotate = False):
    """
    Downsizes an image by an integer ratio.
    Parameters:
        X (np.ndarray[height, width]) : 2D image
        ratio (int) : ratio to downsize with
        rotate (bool) : whether to perform downsizing from all possible starting positions. Defaut: False.
        
    Returns:
        gen (generator) : a generator of downsized images.
    """
    
    if rotate:
        ixs = np.arange(ratio)
        iys = np.arange(ratio)
    else:
        ixs = iys = np.zeros(1, dtype = np.int)
    
    gen = (_shrunk_one(X, ratio, ix, iy) for ix in ixs for iy in iys) 
    
    return gen

    
def _shrunk_one(X, ratio, ix, iy):
    """
    Downsizes an image by an integer ratio.
    Parameters:
        X (np.ndarray[height, width]) : 2D image
        ratio (int) : ratio to downsize with
        ix (int) : start pixel
        iy (int) : start pixel
        
    Returns:
        X_shrunk (np.ndarray[width // 2, height // 2]) : downized image
    """
    
    # @TODO input checks
    
    X_shrunk = X[ix::ratio,iy::ratio]
    
    return X_shrunk


class ImageScalerCleaner:
    
    @property
    def ratio(self):
        return self._ratio
    
    
    @property
    def tol(self):
        return self._tol
    
    @property
    def histo(self):
        return self._histo
        
        
    def __init__(self, ratio = 3, tol = 0.1):
        """
        Parameters:
            ratio (int) : ratio with which the image is downsized. Default: 3.
            tol (float) : tolerance. Default: 0.1.
        """
        
        self._ratio = ratio
        self._tol = tol
        self._histo = None
    
    
    def clean(self, image):
        """
        Replaces satellite colours with main colours.
        Parameters:
            image (np.ndarray[height, width, {3,4}]) : RGB(A) image
            
        Returns:
            cleaned (np.ndarray[height, width, 3]) : cleaned RGB image
        """
    
        # encode image to RRGGBB
        encoded = ImageEncoder.encode(image)
    
        # separate main and satellite colours
        main_colour_codes, satellite_colour_codes = self._find_satellite_colours(encoded)
    
        # assing satellite colours to main colours
        cluster_colour_map = self._assign_to_main_colours(main_colour_codes, satellite_colour_codes)
    
        # applied colour mapping
        applied = apply_colour_clustering_to_flag(encoded, cluster_colour_map)
        
        # decode image to RR,GG,BB format
        cleaned = ImageEncoder.decode(applied)
        
        # calculate histogram of the cleaned image
        self._histo = calculate_colour_histogram(applied)
        
        return cleaned
    
    
    def _find_satellite_colours(self, encoded):
        """
        Parameters:
            encoded (np.ndarray[height, width]) : RRGGBB encoded image 
        
        Returns:
            main_colours (np.ndarray) : codes of main colours
            satellite_colours (np.ndarray) codes of satellite colours
        """
        
        # count colours in original image
        counter1 = Counter(encoded.flat)
        
        
        # loop over all downsized images
        for idx, shrunk in enumerate(_shrunk(encoded, 3, rotate = True)):
        
            # count colours in downsized image
            counter2 = Counter(shrunk.flat)
            
            # only count colours in intersections
            if idx == 0:
                accumulator = counter2
            else:
                accumulator &= counter2
        
        # calculate ratio of colours counts in original and downsized images
        diff = np.array([[k, counter1[k], accumulator[k] * 1.0, k] for k in accumulator])
        diff[:,3] = diff[:,1] / diff[:,2]
        
        # accept those colours as main whose ratio is close enough to the expected
        expected_ratio = self.ratio * self.ratio
        mask = np.abs(diff[:,3] - expected_ratio) < expected_ratio * self.tol
        
        # select main colours, and arrange them in descending order
        main_colour_codes = diff[mask, 0]
        main_colour_codes = main_colour_codes[np.argsort(diff[mask,1][::-1])]
            
        # satellite colours are those which are not main
        satellite_colour_codes = np.array([x for x in counter1.keys() if not x in main_colour_codes.tolist()])
         
        return main_colour_codes, satellite_colour_codes
    
    
    def _assign_to_main_colours(self, main_colour_codes, satellite_colour_codes):
        """
        Assigns satellite colours to main colours based on Voronoi paritioning.
        
        Parameters:
            main_colour_codes (np.ndarray) : colour codes of the main colours
            satellite_colour_codes (np.ndarray) : colour codes of the main colours
            
        Returns:
            cluster_colour_map ({int : np.ndarray}) : colour replacement map
        """
    
        # convert to R,G,B array
        main_colours = np.array([ColourEncoder.decode(x) for x in main_colour_codes])
        satellite_colours = np.array([ColourEncoder.decode(x) for x in satellite_colour_codes])
        
        # create cluster mapping
        
        # there are no satellite colours
        cluster_colour_map = {main_colour_codes[idx] : main_colour_codes[idx] 
            for idx in np.arange(main_colour_codes.size)}
        
        # there are satellite colours
        if satellite_colours.size != 0:
            # find closest main colour (Voronoi again)
            idcs = np.argmin(distance_matrix(satellite_colours, main_colours), axis = 1)
        
            # replace mapping
            for idx in np.arange(main_colour_codes.size):
                cluster_colour_map.update({
                                           main_colour_codes[idx] : 
                                           np.append(satellite_colour_codes[idcs == idx], main_colour_codes[idx])
                                          })
        
        return cluster_colour_map