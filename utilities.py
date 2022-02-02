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
import configparser
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sn

from progress.bar import Bar
import pandas as pd

from fuzzywuzzy import fuzz

# Read the config file for later use
cf = configparser.ConfigParser()
cf.read("config.ini")

def Pickle_write(path, file_name, object):
    destination = path + file_name

    with open(destination, 'wb') as f:
        pickle.dump(object, f)

def Pickle_read(path, file_name):
    destination = path + file_name

    with open(destination, 'rb') as f:
        return pickle.load(f)

def convert_syllable_labels(df):
    # Convert the labels from int to str
    df['length'] = np.where(df['length'] == 0, 'short', df['length'])
    df['length'] = np.where(df['length'] == 1, 'long', df['length'])
    df['length'] = np.where(df['length'] == 2, 'elision', df['length'])
    return df

def clean(ll):

    """Remove all corrupt lines from a set of bs4 <line>s

    Args:
        ll (list of bs4 <line>): Lines to clean

    Returns:
        (list of bs4 <line>): The lines, with the corrupt ones removed.
    """

    return [
        l
        for l in ll
        if l.has_attr("pattern")
        and l["pattern"] != "corrupt"
        and l["pattern"] != "not scanned"
    ]

def Create_files_list(path, substring):
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

def create_line_plot(plots, ylabel, xlabel, plot_titles, title, plotname):
    # Simple function to easily create plots
    path = './plots/'
    time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    full_file_name = '{0}{1}_{2}.png'.format(path, plotname, time)
    
    for plot_line in plots:
        plt.plot(plot_line)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(plot_titles, loc='lower right')
    plt.title(title)
    plt.savefig(full_file_name)
    plt.show()
    # plt.clf()

def create_heatmap(dataframe ,xlabel, ylabel, title, filename, vmin=None, vmax=None, path='./plots/'):
    # Simple function to create a heatmap
    # dataframe.to_numpy().max()
    plt.clf()
    
    path = path
    time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    full_file_name = '{0}{1}_{2}.png'.format(path, time, filename)

    sn.set(font_scale=1.4)
    # sn.heatmap(dataframe, annot=True, fmt='g', annot_kws={"size": 16}, cmap='Blues', vmin=vmin, vmax=vmax)
    sn.heatmap(dataframe, annot=True, fmt='g', cmap='Blues', vmin=vmin, vmax=vmax, cbar=False)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(full_file_name, bbox_inches='tight')        
    
    return plt


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
    merged_list = Pickle_read(path, texts.pop())
    # merge all other texts into this initial list
    for text_list_id in texts:
        # from the list with texts
        text_list = Pickle_read(path, text_list_id)
        # take every sentence and add it to the merged_list
        for sentence_numpy in text_list:
            merged_list.append(sentence_numpy)
    return merged_list     

def restricted_float(x):
    # Used in argparser
    try:
        x = float(x)
    except ValueError:
        print("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        print("%r not in range [0.0, 1.0]"%(x,))
    return x       

def convert_pedecerto_to_sequence_labeling(df) -> list:
    """Converts the given pedecerto dataframe to a list with sequence labels. More specifically,
    one list with multiple lists is returned. Each sublist represents a sentence with syllable and label.
    Such sublist looks as follows: [(syllable, label),(syllable, label), (syllable, label)]

    Args:
        df (dataframe): of a text in the pedecerto format

    Returns:
        list: with sequence labels (to serve as input for sequence labeling tasks)
    """              
    # Create a list to store all texts in
    all_sentences_list = []
    # get the integers for all titles to loop through
    all_titles = df['title'].unique()
    for title in Bar('Converting Pedecerto to CRF').iter(all_titles):
        # Get only lines from this book
        title_df = df.loc[df['title'] == title]
        # Per book, process the lines
        all_lines = title_df['line'].unique()
        for line in all_lines:
            line_df = title_df[title_df["line"] == line]

            length_list = line_df['length'].to_numpy()
            syllable_list = line_df['syllable'].to_numpy()
            # join them into 2d array and transpose it to get the correct crf format:
            combined_list = np.array((syllable_list,length_list)).T
            # Append all to the list which we will return later
            all_sentences_list.append(combined_list)

    return all_sentences_list    

def convert_pedecerto_dataframes_to_sequence_labeling_list(source, destination):
    """Converts all pedecerto dataframes in the given location to sequence labeling lists.
    Saves these to disk in the specified location.

    Args:
        source (string): source location of the pedecerto dataframes
        destination (string): destination location of the pedecerto dataframes
    """        
    texts = Create_files_list(source, '.pickle')
    for text in texts:
        df = Pickle_read(source, text)
        # Convert the integer labels to string labels (like sequence labeling likes)
        df = convert_syllable_labels(df)
        # convert the current file to a sequence labeling list
        sequence_label_list = convert_pedecerto_to_sequence_labeling(df)
        # extract the name of the file to be used for pickle saving
        text_name = text.split('.')[0]
        text_name = text.split('_')[-1]
        # And write it to the location specified
        Pickle_write(destination, text_name, sequence_label_list)

def convert_pedecerto_to_sequence_labeling(df) -> list:
    """Converts the given pedecerto dataframe to a list with sequence labels. More specifically,
    one list with multiple lists is returned. Each sublist represents a sentence with syllable and label.
    Such sublist looks as follows: [(syllable, label),(syllable, label), (syllable, label)]

    Args:
        df (dataframe): of a text in the pedecerto format

    Returns:
        list: with sequence labels (to serve as input for sequence labeling tasks)
    """              
    # Create a list to store all texts in
    all_sentences_list = []
    # get the integers for all titles to loop through
    all_titles = df['title'].unique()
    for title in Bar('Converting Pedecerto to CRF').iter(all_titles):
        # Get only lines from this book
        title_df = df.loc[df['title'] == title]
        # Per book, process the lines
        all_lines = title_df['line'].unique()
        for line in all_lines:
            line_df = title_df[title_df["line"] == line]

            length_list = line_df['length'].to_numpy()
            syllable_list = line_df['syllable'].to_numpy()
            # join them into 2d array and transpose it to get the correct crf format:
            combined_list = np.array((syllable_list,length_list)).T
            # Append all to the list which we will return later
            all_sentences_list.append(combined_list)

    return all_sentences_list

def combine_sequence_label_lists(list_with_file_names, output_name, path):
    """Simple function to combine sequence label lists in pickle format.

    Args:
        list_with_file_names (list): with file names (no extension!) to be processed. Should be in the 
        path_sequence_labels folder.
        output_name (string): destination name to be saved as
    """    
    # Add the pickle extension to our given files
    list_with_file_names = [x+'.pickle' for x in list_with_file_names]    
    # And to the output name
    output_name = output_name + '.pickle'
    merged_list = merge_sequence_label_lists(list_with_file_names, path)
    Pickle_write(cf.get('Pickle', 'path_sequence_labels'), output_name, merged_list)

def get_str_similarity(a, b):
    """ Returns the ratio of similarity between the two given strings

    Args:
        a (str): first string to be compared
        b (str): second string to be compared

    Returns:
        integer: of ratio of similarity between to arguments (value between 0 and 100)
    """     
    # remove punctuation and capitalisation
    # a = a.translate(str.maketrans('', '', string.punctuation)).lower()
    # b = b.translate(str.maketrans('', '', string.punctuation)).lower()
    return fuzz.token_sort_ratio(a,b)   

def find_stem(arr):
    """Finds longest substring in the givven array

    Args:
        arr (list): with strings

    Returns:
        str: of longest common substring
    """    
    # Determine size of the array
    n = len(arr)

    if n == 1:
        return arr[0]

     # Take first word from array
    # as reference
    s = arr[0]
    l = len(s)
    res = ""
    for i in range(l):
        for j in range(i + 1, l + 1):
            # generating all possible substrings
            # of our reference string arr[0] i.e s
            stem = s[i:j]
            k = 1
            for k in range(1, n):
                # Check if the generated stem is
                # common to all words
                if stem not in arr[k]:
                    break
            # If current substring is present in
            # all strings and its length is greater
            # than current result
            if (k + 1 == n and len(res) < len(stem)):
                res = stem
    return res

def auto_combine_sequence_label_lists():
    # Automatically combines Pedecerto files with similar names. For example, OV-amo1 and OV-amo2 will be merged and saved
    # as one pickle :D
    EQUALITY_RATIO = 80 # Ratio between OV-amo1 and OV-amo2 is 86.
    mutable_list = Create_files_list(cf.get('Pedecerto', 'path_xml_files'), 'pickle')
    non_mutable_list = Create_files_list(cf.get('Pedecerto', 'path_xml_files'), 'pickle')

    mutable_list = {x.replace('.pickle', '') for x in mutable_list}
    non_mutable_list = {x.replace('.pickle', '') for x in non_mutable_list}

    processed_list = set()

    for item in non_mutable_list:

        if item in processed_list:
            continue

        current_text_list = [item]
        
        for item2 in mutable_list:

            if item2 in processed_list:
                continue

            if get_str_similarity(item, item2) >= EQUALITY_RATIO:
                current_text_list.append(item2)
                
                processed_list.add(item)
                processed_list.add(item2)
                processed_list = set(processed_list)
        
        current_text_list = list(set(current_text_list))
        file_name = find_stem(current_text_list)
        print(file_name)
        print(current_text_list)

        combine_sequence_label_lists(current_text_list, file_name, cf.get('Pedecerto', 'path_xml_files'))



if __name__ == "__main__":
    
    auto_combine_sequence_label_lists()
    exit(0)

    # Create the Trimeter dataset
    df = pd.read_csv('./texts/iambic/agamemnon_labels_4.csv')
    sequence_label_list = convert_pedecerto_to_sequence_labeling(df)

    new_list = []

    # Remove trailing spaces
    for line in sequence_label_list:
        if line[-1][0] == '-' and line[-1][1] == 'space':
            new_list.append(line[:-1])
        else:
            new_list.append(line)
    
    Pickle_write(cf.get('Pickle', 'path_sequence_labels'), 'SEN-aga.pickle', new_list)    
