import numpy as np

from image_utils import ColourEncoder

class ColourHistogram:
    
    @property
    def colours(self):
        return self._colours
    
    @property
    def counts(self):
        return self._counts
    
    @property
    def codes(self):
        return self._codes
    
    
    def __init__(self, colours, counts, codes):
        
        self._colours = colours
        
        self._counts = counts
        
        self._codes = codes
        
            
def calculate_colour_histogram(encoded_image):
    """
    Creates a histogram of the colours.
    """
    
    if encoded_image.ndim != 2:
        raise ValueError("Image must be 2D")
    
    # histogram colours
    codes, counts = np.unique(encoded_image.flat, return_counts = True)
    counts = counts / counts.sum()
    
    # sort to descending order
    idcs = np.argsort(counts)[::-1]
    codes = codes[idcs]
    counts = counts[idcs]
    
    # convert encoded colours to RGB [0--255] triplets
    colours = np.array([ColourEncoder.decode(x) for x in codes])
    
    histogram = ColourHistogram(colours, counts, codes)
    
    return histogram