import numpy as np


def create_cluster_colour_map(codes, centre_labels, labels):
    """
    Replaces satellite colours with their respective main colours.
    Parameters:
        codes (np.ndarray[n_colours] of int) : list of colour codes
        centre_labels (np.ndarray[n_clusters] of int) indices of the cluster centres
        labels (np.ndarray[n_colours] of int) : labels assigned to each colour
        
    Returns:
        cluster_colour_map ({int : np.ndarray}) : colour replacement map.
    """
    
    cluster_colour_map = {}
    
    # this can be one line, really
    for centre_label in centre_labels:
        mask = labels == centre_label
        cluster_colour_map.update({codes[centre_label] : codes[mask]})
        
    return cluster_colour_map
    

def apply_colour_clustering_to_flag(encoded, cluster_colour_map):
    """
    Applies a colour clustering to a flag. Each satellite colour is replaced by 
    the main colour it belongs to.
    Parameters:
        encoded (np.ndarray[width, height]) : RRGGBB encoded image
        cluster_colour_map ({int : np.ndarray}) : colour replacement map
        
    Returns:
        applied (np.ndarray[width, height]) : RRGGBB encoded image with replaced colours
    """
    
    applied = np.zeros_like(encoded)
    
    for main_colour, satellite_colours in cluster_colour_map.items():
        
        mask = np.isin(encoded, satellite_colours)
        applied[mask] = main_colour
        
    return applied
    

class ColourEncoder:

    @staticmethod
    def encode(colour):
        """
        Converts a base256 encoded colour triplet to a single integer representation.
        """
    
        encoded = colour[0] * 65536 + colour[1] * 256 + colour[2]
    
        return encoded


    @staticmethod
    def decode(colour):
        """
        Converts a single base256 encoded colour triplet to a triplet of [0,255] integers
        """
    
        r = colour // 65536
        g = (colour - r * 65536) // 256
        b = colour % 256
     
        decoded = np.array((r, g, b))
    
        return decoded
        

class ImageEncoder:
    
    @staticmethod
    def encode(X):
        """
        The colours are encoded with a single integer in base256.
        """
        
        if X.ndim != 3:
            raise ValueError("Only 3D images (H,W,(R,G,B,A)) are accepted. Got: {0}".format(X.shape))
    
        encoder = np.array([65536, 256, 1])
        encoded_image = np.dot(X[:,:,:3], encoder)

        return encoded_image
    
    
    @staticmethod
    def decode(X):
        """
        #RRGGBB 256-level image to (R,G,B) image converter
        Parameters:
            X (np.ndarray(height, width) of int): image
        Returns:
            decoded_image (np.ndarray([height, width, 3])) : (R,G,B) image
        """
        
        if X.ndim != 2:
            raise ValueError("Image must be of shape 2. Got: {0}".format(X.shape))
        
        decoded_image = np.zeros(np.concatenate([X.shape, np.array([3])]),
                                 dtype = np.int)
         
        decoded_image[:,:,0] = X // 65536
        decoded_image[:,:,2] = X % 256
        decoded_image[:,:,1] = (X - decoded_image[:,:,0] * 65536) // 256
        
        return decoded_image
        
        
class ImageCompressor:
    
    @staticmethod  
    def compress(X):
        """
        Compresses an 2D image by marking the regions of identical colour.
        """
           
        compressed = []
        
        if image.shape != 2:
            raise ValueError("Image must be a 2D array")
        n_row, ncol = image.shape
            
        # iterator over doublets of pixels
        gen1, gen2 = tee(X.flat)
        
        i_start = 0
        for idx, (x1, x2) in enumerate(zip(gen1, gen2)):
            if x1 != x2:
                compressed.append([x1, i_start, idx -1])
                i_start = idx
        
        # last colour
        compressed.append([x2, i_start, idx])
                                  
        # calculate row and coloumn indices
        compressed = np.array(compressed)
        
        # bundle colours and 
        compressed = np.hstack(
                [
                    compressed[:,0],
                    compressed[:,1] // n_col, 
                    compressed[:,1] % n_col,
                    compressed[:,2] // n_col, 
                    compressed[:,3] % n_col 
                ])
        
        return compressed
                
    @staticmethod
    def decompress(compressed):
        """
        Parameters:
            compressed (np.ndarray) :
                each row [colour, row_start, col_start, row_end, col_end]
        Returns:
            image (np.ndarray[height, width, 3] of int) : decompressed image.
        """
        
        # create blank image
        image = np.zeros((compressed[-1,2], compressed[-1, 3], 3), dtype = np.int)
        
        for colour, row_start, col_start, row_end, col_end in compressed:
            
            rgb = ColourEncoder.decode(colour)
            
            image[row_start, col_start:, :] = rgb
            
            if row_end > row_start + 1:
                image[row_start + 1 : row_end - 1, :, :] = rgb
            
            image[row_end, :col_end, :] = rgb