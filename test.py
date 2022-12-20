from lsnn import dataset_creation
from lsnn.lstm import Latin_LSTM
from lsnn.crf import Latin_CRF

from lsnn import config as conf
from lsnn import utilities as util

####################
# DATASET CREATION #
####################
dataset_creation.Pedecerto_parser().convert_pedecerto_xml_to_syllable_sequence_files(
    input_folder = conf.PEDECERTO_SCANSION_FOLDER,
    output_folder = conf.SEQUENCE_LABELS_FOLDER    
)

dataset_creation.Anceps_parser().convert_anceps_json_to_syllable_sequence_files(
    input_folder = conf.ANCEPS_SCANSION_FOLDER,
    output_folder = conf.SEQUENCE_LABELS_FOLDER
)

util.combine_sequence_label_lists(
    list_with_file_names = util.create_files_list(conf.SEQUENCE_LABELS_FOLDER, 'pickle'), 
    output_name = 'combined.txt', 
    destination_path = conf.SEQUENCE_LABELS_FOLDER,
    add_extension = False
)

#############
# CRF MODEL #
#############
latin_crf = Latin_CRF()

result = latin_crf.custom_prediction(
    predictor = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'ENN-anna.pickle'),
    predictee = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HOR-arpo.pickle')
)
print(result)

result = latin_crf.kfold_model(
    sequence_labels = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'ENN-anna.pickle'),
    splits = 5
)
print(result)

##############
# LSTM MODEL #
##############
lstm = Latin_LSTM(
    sequence_labels_folder = conf.SEQUENCE_LABELS_FOLDER,
    models_save_folder = conf.MODELS_SAVE_FOLDER,
    anceps_label = False,
) 

model = lstm.create_model(
    text = 'HEX_ELE-all.pickle', 
    num_epochs = 2,
    save_model = True, 
    model_name = 'temp'
)

model = lstm.load_model(
    path = conf.MODELS_SAVE_FOLDER + 'HEX_ELE-all'
)

# with the model, we can predict the labels of a given set
test_set = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle')
result = lstm.predict_given_set(test_set, model)