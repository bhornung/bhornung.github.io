import json

def load_from_json(path_to_db):
    """
    Loads the contents of a json file.
    Parameters:
        path_to_db (str) : full path to a json file.
    Returns:
        data_ (object) : the loaded object.
    """
    with open(path_to_db, 'r') as fproc:
        data_ = json.load(fproc)
            
    return data_
    
    
def load_dict_from_json(path_to_db, convert_keys_to_int = False):
    """
    Loads the contents of a json file.
    Parameters:
        path_to_db (str) : full path to a json file.
    Returns:
        data_ (object) : the loaded object.
    """
    with open(path_to_db, 'r') as fproc:
        dict_ = json.load(fproc)

    if not isinstance(dict_, dict):
        raise TypeError("Loaded object is not a dictionary.")
            
    try:
        dict_ = {int(k) : v for k, v in dict_.items()}
    except:
        raise
    
    return dict_