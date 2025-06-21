import json
import os
import pickle

def format_string(string: str) -> str:
    """
    Simple function to strip punctuation and change j to i
    """
    string: str = ''.join(c for c in string if c.isalnum() or c.isspace())
    string = string.replace("J", "I")
    string = string.replace("j", "i")
    string = string.lower()

    return string

def retrieve_length(list: list) -> str: 
    """
    Retrieves the length from the given class list
    """
    if "long" in list:
        return "long"
    elif "short" in list:
        return "short"
    elif "elided" in list:
        return "elision"
    elif "resolved" in list:
        return "short"
    else:
        return 'corrupt'

def write_pickle(filename: str, variable) -> None:
    with open(filename, 'wb') as file:
        pickle.dump(variable, file)

def read_pickle(file: str):
    with open(file, 'rb') as file:
        return pickle.load(file)

def get_files_list(path: str) -> list:
    return os.listdir(path)
