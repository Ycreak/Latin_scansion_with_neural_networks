'''
       ___       ___           ___     
      /\__\     /\  \         /\__\    
     /:/  /    /::\  \       /::|  |   
    /:/  /    /:/\ \  \     /:|:|  |   
   /:/  /    _\:\~\ \  \   /:/|:|__|__ 
  /:/__/    /\ \:\ \ \__\ /:/ |::::\__\ 
  \:\  \    \:\ \:\ \/__/ \/__/~~/:/  /
   \:\  \    \:\ \:\__\         /:/  / 
    \:\  \    \:\/:/  /        /:/  /  
     \:\__\    \::/  /        /:/  /   
      \/__/     \/__/         \/__/    
'''
# Latin Scansion Model
# Philippe Bors and Luuk Nolden
# Leiden University 2021

# Library Imports
import numpy as np
from progress.bar import Bar
import argparse

# Class Imports
# from word2vec import Word_vector_creator 
# from preprocessor import Text_preprocessor 
# from neuralnetwork import Neural_network_handler

import utilities as util

p = argparse.ArgumentParser(description='Argument parser for the LatinScansionModel')
p.add_argument('--pedecerto_conversion', action="store_true",
                help='converts the pedecerto XML files stored in pedecerto/xml_files into labeled dataframes')
p.add_argument('--sequence_labels_conversion', action="store_true", 
               help='converts the pedecerto dataframes stored in pedecerto/df_pedecerto into sequence labeling lists')
p.add_argument('--combine_author_files', action="store_true", 
               help='')
p.add_argument('--create_hexameter_set', action="store_true", 
               help='')
p.add_argument('--create_elegiac_set', action="store_true", 
               help='')               
p.add_argument('--create_hexameter_elegiac_set', action="store_true", 
               help='')
FLAGS = p.parse_args()

''' This functionality will take all pedecerto XML files stored in ./pedecerto/xml_files and turn them into
pandas dataframes with the columns author, title, book, line, syllable, length. This information can be used
as the ground truth for the neural network models in the rest of this program. The converted files will be stored
in pickle format in ./pedecerto/pedecerto_df. Processed files will be moved to the processed subdirectory.
'''
if FLAGS.pedecerto_conversion:
    print('Converting pedecerto XML into sequence label lists')
    from pedecerto.textparser import Pedecerto_parser
    pedecerto_parse = Pedecerto_parser(source = util.cf.get('Pedecerto', 'path_xml_files'),
                                       destination = util.cf.get('Pedecerto', 'path_xml_files'))

if FLAGS.combine_author_files:
    # Combine the following lists
    util.auto_combine_sequence_label_lists()

    # tibullus = ['TIB-ele1', 'TIB-ele2', 'TIB-ele3']
    # util.combine_sequence_label_lists(tibullus, 'TIB-ele', util.cf.get('Pickle', 'path_sequence_labels'))

    # propertius = ['PROP-ele1', 'PROP-ele2', 'PROP-ele3', 'PROP-ele4']
    # util.combine_sequence_label_lists(propertius, 'PROP-ele', util.cf.get('Pickle', 'path_sequence_labels'))

    # ovidius = ['OV-amo1', 'OV-amo2', 'OV-amo3', 'OV-epis', 'OV-fast', 'OV-ibis', 'OV-medi', 'OV-pon1', 'OV-pon2',
    #            'OV-pon3', 'OV-pon4', 'OV-tri1', 'OV-tri2', 'OV-tri3', 'OV-tri4', 'OV-tri5']
    # util.combine_sequence_label_lists(ovidius, 'OV-ele', util.cf.get('Pickle', 'path_sequence_labels'))   

# Provide which texts should be used as the hexameter and elegiac train sets
hexameter_texts = ['VERG-aene', 'IVV-satu', 'LVCR-rena', 'OV-meta', 'PERS-satu']
elegiac_texts = ['PROP-ele', 'OV-ele']

if FLAGS.create_hexameter_set:
    util.combine_sequence_label_lists(hexameter_texts, 'HEX-all', util.cf.get('Pickle', 'path_sequence_labels'))

if FLAGS.create_elegiac_set:
    util.combine_sequence_label_lists(elegiac_texts, 'ELE-all', util.cf.get('Pickle', 'path_sequence_labels'))

if FLAGS.create_hexameter_elegiac_set:
    hexameter_elegiac = hexameter_texts + elegiac_texts
    util.combine_sequence_label_lists(hexameter_elegiac, 'HEX_ELE-all', util.cf.get('Pickle', 'path_sequence_labels'))



# temp = util.Pickle_read(util.cf.get('Pedecerto', 'path_xml_files'), 'VERG-aene.pickle')
# print(temp[:3])

exit(0)

########
# MAIN #
########
class Vector:
    # Wrapper class for vectors in a pandas dataframe 
    def __init__(self, v):
        self.v = v

# Parameters to run each step
run_preprocessor = False
run_pedecerto = False
run_model_generator = False
add_embeddings_to_df = False 
run_neural_network = False

''' Run the preprocessor on the given text if needed.
This reads the text, cleans it and returns a list of syllables for now
To achieve this, the pedecerto tool is used
'''
if run_preprocessor:
    print('Running preprocessor')
    preprocessor = Text_preprocessor(util.cf.get('Text', 'name'))
    util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'char_list'), preprocessor.character_list)

# Load the preprocessed text
character_list = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'char_list'))
if int(util.cf.get('Util', 'verbose')): print(character_list)

''' Now create a dataframe. Containing: syllable, length, vector.
'''
if run_pedecerto:
    print('Running pedecerto parser')
    parse = Pedecerto_parser(util.cf.get('Pedecerto', 'path_texts'))
    # This function created pickle files for all texts that are in the ./texts/ folder

pedecerto_df = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'pedecerto_df'))
if util.cf.get('Util', 'verbose'): print(pedecerto_df)

# Run the model generator on the given list if needed
if run_model_generator:
    print('Running Word2Vec model generator')
    # Create a word2vec model from the provided character list
    word2vec_creator = Word_vector_creator(character_list, util.cf.getint('Word2Vec', 'vector_size'), util.cf.getint('Word2Vec', 'window_size'))
    util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'word2vec_model'), word2vec_creator.model)

# Load the saved/created model
word2vec_model = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'word2vec_model'))

# Add the embeddings created by word2vec to the dataframe
if add_embeddings_to_df:
    print('Adding embeddings to the dataframe')

    df = pedecerto_df
    df['vector'] = df['syllable']

    # Add syllable vectors to the dataframe using the word2vec model
    unique_syllables = set(df['syllable'].tolist())

    for syllable in Bar('Processing').iter(unique_syllables):
        try:
            vector = word2vec_model.wv[syllable]
            # Pump (wrapped) vector to the applicable positions
            df['vector'] = np.where(df['vector'] == syllable, Vector(vector), df['vector'])
        except:
            IndexError('Syllable has no embedding yet.')

    util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'embedding_df'), df)

# Provide the neural network with the dataframe
if run_neural_network:
    print('Running the neural network generation')

    df = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'embedding_df'))
    if util.cf.get('Util', 'verbose'): print(df)

    nn = Neural_network_handler(df)





