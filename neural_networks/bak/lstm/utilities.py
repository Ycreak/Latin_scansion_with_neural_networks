import seaborn as sn
import matplotlib.pyplot as plt

def flatten_list(given_list: list) -> list:
    """Flattens the given list. Meaning that nested lists inside the given lists are
    turned into one list. For example, [[a,b],[c]] -> [a,b,c]

    Args:
        given_list (list): nested list that needs to be flattened

    Returns:
        list: flattened list
    """        
    return [item for sublist in given_list for item in sublist]

def merge_sequence_label_lists(texts, path):
    """Merges the given lists (contained in sequence labeling pickles) on the given path.
    Outputs one list with all sentences of the given texts in sequence labeling format.
    Useful when merging all metamorphoses for example.

    Args:
        texts (list): of sequence labeling pickled files
        path (string): where these pickled files are stored

    Returns:
        list: of merged texts
    """        
    # Create a starting list from the last entry using pop
    merged_list = pickle_read(path, texts.pop()) #FIXME: is this a call by reference?
    # merge all other texts into this initial list
    for text_list_id in texts:
        # from the list with texts
        text_list = pickle_read(path, text_list_id)
        # take every sentence and add it to the merged_list
        for sentence_numpy in text_list:
            merged_list.append(sentence_numpy)
    return merged_list         

def create_heatmap(dataframe, xlabel, ylabel, title, filename):
    # Simple function to create a heatmap
    sn.set(font_scale=1.4)
    sn.heatmap(dataframe, annot=True, fmt='g', annot_kws={"size": 16}, cmap='Blues')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')        
    plt.clf()    
