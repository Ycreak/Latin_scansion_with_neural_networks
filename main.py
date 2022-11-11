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
import argparse
import utilities as util

p = argparse.ArgumentParser(description='Argument parser for the LatinScansionModel')

# Parameters for the dataset creation
p.add_argument('--pedecerto_conversion', action="store_true",
                help='converts the pedecerto XML files stored in pedecerto/xml_files into labeled dataframes')
p.add_argument('--sequence_labels_conversion', action="store_true", 
               help='converts the pedecerto dataframes stored in pedecerto/df_pedecerto into sequence labeling lists')
p.add_argument('--combine_author_files', action="store_true", 
               help='')
p.add_argument('--output_name', default='combined_syllable_label_set', type=str, 
               help='')               
p.add_argument('--create_hexameter_set', action="store_true", 
               help='')
p.add_argument('--create_elegiac_set', action="store_true", 
               help='')               
p.add_argument('--create_hexameter_elegiac_set', action="store_true", 
               help='')

# Parameters for the models
p.add_argument('--custom_train_test', action="store_true", 
                help='Run the model with the provided train and test set')
p.add_argument('--kfold', action="store_true", 
                help='Train and test the model on the same training set using kfold cross validation')
p.add_argument('--train', default='none', type=str, 
                help='Dataset to train the model on')
p.add_argument('--test', default='none', type=str, 
                help='Dataset to test the model on')

# Parameters specific for the CRF model
p.add_argument('--crf', action="store_true", 
                help='Run the CRF model with the given parameters')

# Parameters specific for the LSTM model
p.add_argument('--lstm', action="store_true", 
                help='Run the LSTM model with the given parameters')
p.add_argument("--create_model", action="store_true", 
                help="specify whether to create the model: if not specified, we load from disk")
p.add_argument("--save_model", action="store_true", 
                help="specify whether to save the model: if not specified, we do not save")
p.add_argument("--verbose", action="store_true", 
                help="specify whether to run the code in verbose mode")
p.add_argument('--epochs', default=25, type=int, 
                help='number of epochs')

FLAGS = p.parse_args()

''' TODO:
'''
if FLAGS.pedecerto_conversion:
    print('Converting pedecerto XML into sequence label lists')
    from pedecerto.textparser import Pedecerto_parser
    pedecerto_parse = Pedecerto_parser(source = util.cf.get('Pedecerto', 'path_xml_files'),
                                       destination = util.cf.get('Pedecerto', 'path_xml_files'))

''' TODO:
'''
if FLAGS.combine_author_files:
    # Combine the following lists
    # util.auto_combine_sequence_label_lists()

    combined_folder = util.cf.get('Pedecerto', 'path_xml_files') + 'combine'
    entries = util.Create_files_list(combined_folder, 'pickle')
    print(entries)

    if(entries):
        util.combine_sequence_label_lists(entries, FLAGS.output_name, util.cf.get('Pickle', 'path_sequence_labels'), add_extension=False) 

if FLAGS.crf:
    from crf import CRF_sequence_labeling
    crf_result = CRF_sequence_labeling(FLAGS)

if FLAGS.lstm:
    from lstm import LSTM_model
    lstm_result = LSTM_model(FLAGS)






