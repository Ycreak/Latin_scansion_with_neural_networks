#       ___       ___           ___     
#      /\__\     /\  \         /\__\    
#     /:/  /    /::\  \       /::|  |   
#    /:/  /    /:/\ \  \     /:|:|  |   
#   /:/  /    _\:\~\ \  \   /:/|:|__|__ 
#  /:/__/    /\ \:\ \ \__\ /:/ |::::\__\
#  \:\  \    \:\ \:\ \/__/ \/__/~~/:/  /
#   \:\  \    \:\ \:\__\         /:/  / 
#    \:\  \    \:\/:/  /        /:/  /  
#     \:\__\    \::/  /        /:/  /   
#      \/__/     \/__/         \/__/    

# Latin Scansion Model
# Philippe Bors and Luuk Nolden
# Leiden University 2021

import pickle
import json

def write_json(given_object: object, file: str) -> None:
    """Writes the given object to the cache

    Args:
        object (object): to store as json
        file (str): path + filename
    """        
    # Serializing json
    json_object = json.dumps(given_object, indent=2)
    # Writing to sample.json
    with open(file, "w") as outfile:
        outfile.write(json_object)

def read_json(file: str) -> dict:
    """Reads the requested object from the cache

    Args:
        file (str): path + filename
    """        
    with open(file, 'r') as openfile:
        # Reading from json file
        return json.load(openfile)

def write_pickle(filename: str, variable: any) -> None:
    with open(filename, 'wb') as file:
        pickle.dump(variable, file)

def read_pickle(file: str):
    with open(file, 'rb') as file:
        return pickle.load(file)

def pickle_write(path, file_name, object):
    destination = path + file_name

    with open(destination, 'wb') as f:
        pickle.dump(object, f)

def pickle_read(path, file_name):
    destination = path + '/' + file_name

    with open(destination, 'rb') as f:
        return pickle.load(f)

def create_files_list(path, substring):
    """Creates a list of files to be processed

    Args:
        path (string): folder to be searched
        substring (string): substring of files to be searched

    Returns:
        list: list with files to be searched
    """
    import os
    
    list = []

    for file in os.listdir(path):
        if file.find(substring) != -1:
            list.append(file)    

    return list
