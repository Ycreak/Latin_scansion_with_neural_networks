from lsnn import dataset_creation
from lsnn.lstm import Latin_LSTM
from lsnn.crf import Latin_CRF

from lsnn import config as conf
from lsnn import utilities as util

####################
# DATASET CREATION #
####################
latin_crf = Latin_CRF()

result = latin_crf.kfold_model(
    sequence_labels = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'VERG-aene.pickle'),
)
print(result)
