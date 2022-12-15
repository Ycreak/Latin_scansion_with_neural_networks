from dataset_creation import Anceps_parser, Pedecerto_parser
from lstm import Latin_LSTM

import config as conf
import utilities as util

# First, convert the anceps files
# Pedecerto_parser().convert_pedecerto_xml_to_syllable_sequence_files()

lstm = Latin_LSTM(
    sequence_labels_folder = conf.SEQUENCE_LABELS_FOLDER,
    models_save_folder = conf.MODELS_SAVE_FOLDER,
    anceps_label = False,
) 

model = lstm.load_model(
    path = conf.MODELS_SAVE_FOLDER + 'HEX_ELE-all'
)

model = lstm.create_model(
    num_epochs = 2,
    text = 'HEX_ELE-all.pickle', 
    save_model = True, 
    model_name = 'temp'
)

test_set = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle')
result = lstm.predict_given_set(test_set, model)
