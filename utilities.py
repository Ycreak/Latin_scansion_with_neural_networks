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

import argparse
# import configparser
# from datetime import datetime
import pickle

import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sn
# from progress.bar import Bar
import pandas as pd



# Read the config file for later use
# cf = configparser.ConfigParser()
# cf.read("config.ini")

def pickle_write(path, file_name, object):
    destination = path + file_name

    with open(destination, 'wb') as f:
        pickle.dump(object, f)

def pickle_read(path, file_name):
    destination = path + '/' + file_name

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

def clean_extra(ll):

    """Remove all corrupt lines from a set of bs4 <line>s, but also those that are uncertain

    Args:
        ll (list of bs4 <line>): Lines to clean

    Returns:
        (list of bs4 <line>): The lines, with the corrupt ones removed.
    """
    temp = []

    for l in ll:
        if l.has_attr("feature"):
            if l["feature"] != "spondaic":
                temp.append(l)
        else:
            temp.append(l)

    ll = temp

    ll = [l for l in ll if l.has_attr("pattern")
        and l["pattern"] != "corrupt"
        and l["pattern"] != "not scanned"
        and l["pattern"] != "SSSS"
    ]


    # for l in ll:
    #     print(l, '\n')

    return ll


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

def merge_lists(list1, list2):
    """Merge two lists into one -> [(l1_a, l2_a), (l1_b, l2_b)]

    Args:
        list1 (list): [description]
        list2 (list): [description]

    Returns:
        list: spliced together
    """    
    return list(zip(list1, list2)) 

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
    texts = create_files_list(source, '.pickle')
    for text in texts:
        df = pickle_read(source, text)
        # Convert the integer labels to string labels (like sequence labeling likes)
        df = convert_syllable_labels(df)
        # convert the current file to a sequence labeling list
        sequence_label_list = convert_pedecerto_to_sequence_labeling(df)
        # extract the name of the file to be used for pickle saving
        text_name = text.split('.')[0]
        text_name = text.split('_')[-1]
        # And write it to the location specified
        pickle_write(destination, text_name, sequence_label_list)

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

def combine_sequence_label_lists(list_with_file_names, output_name, path, add_extension=True):
    """Simple function to combine sequence label lists in pickle format.

    Args:
        list_with_file_names (list): with file names (no extension!) to be processed. Should be in the 
        path_sequence_labels folder.
        output_name (string): destination name to be saved as
    """    
    # Add the pickle extension to our given files
    if add_extension:
        list_with_file_names = [x+'.pickle' for x in list_with_file_names]    
        output_name = output_name + '.pickle'
    # And to the output name
    merged_list = merge_sequence_label_lists(list_with_file_names, path)
    pickle_write(path, output_name, merged_list)

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

def merge_kfold_reports(report_list):
    """This function merges a list of metrics report lists into one. Used to merge several kfold
    metric reports into one final averaged report

    Args:
        report_list (list): with metric reports

    Returns:
        dict: with one final averaged report
    """     
    result_dict = {
        'short': {'precision':0, 'recall':0, 'f1-score':0, 'support':0},
        'elision': {'precision':0, 'recall':0, 'f1-score':0, 'support':0},
        'long': {'precision':0, 'recall':0, 'f1-score':0, 'support':0},
        'weighted avg': {'precision':0, 'recall':0, 'f1-score':0, 'support':0},
    }

    keys = ['long', 'short', 'elision', 'weighted avg']

    # Merge the reports one by one
    for current_dict in report_list:
        for key in keys:
            result_dict[key] = merge_dicts(result_dict[key], current_dict[key])

    # Now divide all values by the number of reports that came in
    for dict in result_dict:
        for key in result_dict[dict]:
            result_dict[dict][key] /= len(report_list)

    return pd.DataFrame(result_dict).T

def merge_dicts(dict1, dict2):
    """Merges two dictionaries using identical keys. Results are added

    Args:
        dict1 (dict): first dictionary to be merged
        dict2 (dict): second dictionary to be merged

    Returns:
        dict: merged dictionary
    """        
    return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in dict1.keys() | dict2.keys()}  

def auto_combine_sequence_label_lists():
    # Automatically combines Pedecerto files with similar names. For example, OV-amo1 and OV-amo2 will be merged and saved
    # as one pickle :D
    from fuzzywuzzy import fuzz
    EQUALITY_RATIO = 80 # Ratio between OV-amo1 and OV-amo2 is 86.
    mutable_list = create_files_list(cf.get('Pedecerto', 'path_xml_files'), 'pickle')
    non_mutable_list = create_files_list(cf.get('Pedecerto', 'path_xml_files'), 'pickle')

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
    
    p = argparse.ArgumentParser()
    p.add_argument("--auto_seq_labels", action="store_true", help="specify whether to create the model: if not specified, we load from disk")
    p.add_argument("--create_trimeter_set", action="store_true", help="specify whether to save the model: if not specified, we do not save")
    p.add_argument("--combine_seq_labels", action="store_true", help="specify whether to run the single text LSTM function")
    p.add_argument("--investigate", action="store_true", help="specify whether to run the hexameter LSTM experiment")
    p.add_argument("--exp_transfer", action="store_true", help="specify whether to run the hexameter transerability LSTM experiment")
    p.add_argument("--exp_elegiac", action="store_true", help="specify whether to run the hexameter genre LSTM experiment")
    p.add_argument("--exp_train_test", action="store_true", help="specify whether to run the train/test split LSTM experiment")
    p.add_argument("--exp_transfer_boeth", action="store_true", help="specify whether to run the Boeth LSTM experiment")
    p.add_argument("--csv2list", action="store_true", help="convert csv into sequence label list")

    p.add_argument("--heatmap", action="store_true", help="specify whether to run the heatmap experiment")
    p.add_argument("--length", action="store_true", help="specify whether to run the heatmap experiment")

    FLAGS = p.parse_args()  

    # text = pickle_read(cf.get('Pickle', 'path_sequence_labels'), 'IVV-satu.pickle')
    # print(len(text))

    if FLAGS.length:
        # Quickly calculate lengths of texts (result is number of verses)
        text = pickle_read('./pickle/sequence_labels/trimeter/', 'Medea.pickle')
        print(len(text))

    if FLAGS.heatmap:

        heatmap_data = pd.read_csv('./csv/elegiac_f1-scores_elision.csv').set_index('predictor')

        # heatmap_data


        # heatmap_data['mean'] = heatmap_data.mean(axis=1)

        # print(heatmap_data)
        # exit(0)

        myplot = create_heatmap(dataframe = heatmap_data,
                        xlabel = 'Test',
                        ylabel = 'Train',
                        title = '{0}: Long f1-scores'.format('hello'),
                        filename = '{0}-long'.format('temp'),
                        path = './plots/experiments/')


    if FLAGS.investigate:
        text = pickle_read(cf.get('Pickle', 'path_sequence_labels'), 'VERG-aene.pickle')
        print(len(text))
        exit(0)
        for line in text:
            for tuple in line:
                if tuple[1] == '':
                    print(line)
                    exit(0)
                    
        
        # print(text[0])

    if FLAGS.combine_seq_labels:
        # combine_sequence_label_lists()
        # trimeter = ['Phoenissae', 'Phaedra', 'Hiempsal', 'Hercules_furens', 'Troades', 'Thyestes', 'Procne', 'Oedipus', 'Octavia', 'Medea', 'Hercules_Oetaeus', 'Ecerinis', 'Agamemnon', 'Achilles']
        trimeter = ['Phoenissae', 'Phaedra', 'Hercules_furens', 'Troades', 'Thyestes', 'Oedipus', 'Octavia', 'Medea', 'Hercules_Oetaeus', 'Agamemnon']
        trimeter = ['Phoenissae', 'Phaedra', 'Hercules_furens', 'Troades', 'Thyestes', 'Oedipus', 'Octavia', 'Medea', 'Hercules_Oetaeus']
        combine_sequence_label_lists(trimeter, 'SEN-proofread', cf.get('Pickle', 'path_sequence_labels'))

    if FLAGS.auto_seq_labels:
        auto_combine_sequence_label_lists()

    if FLAGS.csv2list:
        # input
        input_file = './texts/tetrameter.csv'
        output_file = 'tetrameter.pickle'

        df = pd.read_csv(input_file)
        sequence_label_list = convert_pedecerto_to_sequence_labeling(df)

        pickle_write(cf.get('Pickle', 'path_sequence_labels'), output_file, sequence_label_list)  

    if FLAGS.create_trimeter_set:
        # Create the Trimeter dataset
        df = pd.read_csv('./texts/iambic/agamemnon_labels_6.csv')

        # for i in range(len(df)):
        #     # print(df['anceps'][i])
        #     if df['anceps'][i] == 'space':
        #         df['length'][i] = 'space'

        # df.to_csv('./texts/iambic/agamemnon_labels_5.csv', index=False, header=True)

        # print(df)
        # exit(0)

        sequence_label_list = convert_pedecerto_to_sequence_labeling(df)

        new_list = []

        # Remove trailing spaces
        for line in sequence_label_list:
            if line[-1][0] == '-' and line[-1][1] == 'space':
                new_list.append(line[:-1])
            else:
                new_list.append(line)
        
        # for line in new_list:
        #     print(len(line))
        

        print(new_list)

        # exit(0)
        pickle_write(cf.get('Pickle', 'path_sequence_labels'), 'SEN-aga2.pickle', new_list)    


        # To get some stats about scansions
        # text = util.pickle_read(util.cf.get('Pickle', 'path_sequence_labels'),'SEN-precise.pickle')
        # print(len(text))

        # # new_precise = []
        # # for line in text:
        # #     new_precise.append(line[:-1])

        # exit(0)

        # from collections import Counter

        # result = Counter()
        # for line in text:
        #     temp = Counter(elem[1] for elem in line)
        
        #     result += temp #print(result)

        # print(result)

    def convert_latin_library_text():
        path = './trimeter/text/latinlibrary/'

        text_list = self.create_files_list(path, 'txt')


        for text in text_list:
            new_file = './trimeter/text/' + text
            g = open(new_file, 'w')
            file = path + text

            f = open(file, "r")
            for line in f.readlines():
                # Clean line from non-alpha numeric characters
                line = line.translate({ord(c): None for c in '(),.!?—;:*'})
                # Remove everything between brackets
                line = re.sub(r'\[[^)]*\]', '', line)
                # Remove numbers
                line = ''.join([i for i in line if not i.isdigit()])
                # Strip whitespaces
                line = line.strip().lower()
                line = line + '\n'

                g.write(line)

                print(line)
                # exit(0)
            g.close()
            f.close()
        # print(text_list)