import numpy as np
from skimage.segmentation import felzenszwalb

from histo_utils import ColourHistogram
from image_utils import ColourEncoder, ImageEncoder


class ImageFelzenCleaner:
    """
    Removes satellite colours from and RGB image using Felzenszwalb segmenter.
    """
    
    @property
    def min_size(self):
        return self.min_size
    
    
    @property
    def segmenter_kwargs(self):
        return self._segmenter_kwargs
            
    
    @property
    def histo(self):
        return self._histo
    
    
    def __init__(self, min_size = 'auto', segmenter_kwargs = {}):
        """
        Parameters:
            min_size ({int, 'auto', 'diag'}) : sets the minimum segment size:
                int : use this integer
                auto : use Felzenszwalb's default
                diag : use the length of the diagonal of the image.
            
            segmenter_kwargs ({}) : any additional kwargs for the segmenter
        """
        
        if isinstance(min_size, int):
            self._min_size = min_size
            
        elif isinstance(min_size, str):
            
            if min_size.lower() == 'auto':
                self._min_size = 20 # this is dirty
                
            elif min_size.lower() == 'diag':
                self._min_size = 'diag'
                
            else:
                raise ValueError("min_size must be: int, 'auto', 'diag'. Got: {0}".format(min_size))
        
        else:
            raise ValueError("min_size must be: int, float, 'auto', 'diag'. Got: {0}".format(min_size))
            
        self._segmenter_kwargs = segmenter_kwargs
        
        self._histo = None
        
                
    def clean(self, image):
        """
        Removes satellite colours by applying image segmentation.
        Parameters:
            image (np.ndarray[height, width, {3,4}) : R,G,B,A image

        Returns:
            cleaned(np.ndarray[height, width, 3) : cleaned R,G,B image
        """
          
        if self._min_size == 'diag':
            min_size = int(np.sqrt(image.shape[0] * image.shape[0] + image.shape[1] * image.shape[1]))
        else:
            min_size = self._min_size
            
        self._segmenter_kwargs.update({'min_size' : min_size})
        
        segment_mask = felzenszwalb(image[:,:,:3], **self.segmenter_kwargs)
        
        cleaned, histo = self._clean(image, segment_mask)
        
        self._histo = histo
        
        return cleaned
    
        
    def _clean(self, image, segment_mask):
        """
        i) cleans the image
        ii) creates a histogram
        """
        
        # encode image
        encoded = ImageEncoder.encode(image)
        
        # cleaned encoded image
        encoded_cleaned = np.zeros_like(encoded)
        
        # find unique segments <-- each segment is contigous array of the same integers
        segment_ids = np.unique(segment_mask)
        
        # variables for histogram
        colour_collector = {}
                
        # loop over each segment
        for i_aux, i_seg in enumerate(segment_ids):
            
            # choose one segment
            mask = segment_mask == i_seg
            
            # determine main colour in segment
            colours, counts = np.unique(encoded[mask].flat, return_counts = True)
            main_colour = colours[np.argmax(counts)]

            # replace all colour by main colour
            encoded_cleaned[mask] = main_colour
            
            # store information for histogram
            if not main_colour in colour_collector:
                colour_collector.update({main_colour : mask.sum()})
            else:
                colour_collector[main_colour] += mask.sum()
            
        # transform to R,G,B space
        cleaned = ImageEncoder.decode(encoded_cleaned)
        
        # create histogram
        main_colour_counts, main_colour_codes = \
            np.array(sorted(zip(colour_collector.values(), colour_collector.keys()), reverse = True)).T
            
        main_colour_counts = main_colour_counts / main_colour_counts.sum()
        main_colours = np.array([ColourEncoder.decode(x) for x in main_colour_codes])
        
        histo = ColourHistogram(main_colours, main_colour_counts, main_colour_codes)
        
        return cleaned, histo