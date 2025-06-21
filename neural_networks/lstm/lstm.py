import config as conf
import utilities as util
import models.lstm.utilities as lstm_util

from models.lstm.experiments import make_hexameter_heatmap

from unidecode import unidecode

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import tensorflow as tf
# import tensorflow_addons as tfa


# Based on https://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/

class LSTM():
    """This class provides the user with tools to create an LSTM model for scanning Latin. On startup,
    it prepares the padding procedures and creates syllable and label dictionaries for the one-hot encoding
    used by the LSTM. Having done that, the class can create and save models and predict custom syllable-label lists.
    """    
    PADDING: str = 'padding'
    LABELS: list = ['long', 'short', 'elision', 'space', PADDING]

    def __init__(self):
        # To make the LSTM with integer hashing working, we need to make a list of all syllables from all the texts we are looking at 
        # First, find all our sequence labels files 
        all_sequence_labels_files: list = []
        all_sequence_labels_files += [conf.LSTM_SEQUENCE_LABELS_PATH + '/' + s for s in util.create_files_list(conf.LSTM_SEQUENCE_LABELS_PATH, 'json')] 
        # Merge them into one big file list
        all_sequence_labels: list = self._merge_sequence_label_files(all_sequence_labels_files) 
        # Retrieve all unique syllables from this big list
        # unique_syllables: list = 
        all_syllables = [syllable_tuple[0] for sentence in all_sequence_labels for syllable_tuple in sentence]
        self.unique_syllables: list[str] = list(set(all_syllables)) + [self.PADDING]
        # We need to extract the max sentence length over all these texts to get the padding correct later
        self.max_sentence_length: int = self._retrieve_max_sentence_length(all_sequence_labels)       
        # And we need to create a list of all unique syllables for our word2idx one-hot encoding
        self.word2idx, self.label2idx = self._create_idx_dictionaries(self.unique_syllables, self.LABELS)            

        # With that out of the way, we can start the LSTM process
        print('Preprocessing done.')
        print('number of syllables: ', len(self.unique_syllables))
        print('number of labels: ', len(self.LABELS))
        print('max_sentence_length: ', self.max_sentence_length)
    
    def create_model(self, 
        text : str, 
        num_epochs : int,
        save_model : bool = True,
        model_name : str = 'default'
    ):
        """Function to create a model given a text (pickled sequence label file). Returns the
        created model. In addition, allows the model to be saved to disk.

        Args:
            text (string): name of pickled sequence label file to be used for training
            save_model (bool, optional): whether the created model is to be saved to disk. Defaults to True.
            model_name (str, optional): name to be given to the saved model (folder name). Defaults to 'default'.

        Returns:
            tensorflow model: containing the trained LSTM
        """        
        sequence_labels_training_text = util.read_json(f"{conf.LSTM_SEQUENCE_LABELS_PATH}/{text}")['lines']
        
        X_train, y_train = self.create_X_y_sets(
            sequence_labels_training_text, 
            self.word2idx, 
            self.label2idx, 
            self.max_sentence_length
        )

        model = self._get_model()

        model, history = self._fit_model(
            model = model,
            X = X_train,
            y = y_train,
            batch_size = 32,
            epochs = num_epochs,
            split = 0.2,
            verbose = True
        )   

        if save_model: 
            model.save(f"{conf.LSTM_MODELS_PATH}/{model_name}.keras")

        return model

    def do_prediction(self, 
        model, 
        X: list, 
        y: list
        ):
        """Does a prediction given the model, X and y sets

        Args:
            model (tensorflow model): that needs to do the predicting
            X (list): with syllables that need predicting
            y (list): with labels that are the predictions
        """        
        y_pred = model.predict(X)

        # generate y_pred and return per syllable the integer (syllable) with the highest confidence            
        y_pred = np.argmax(y_pred, axis=-1)
        y = np.argmax(y, axis=-1)

        return y, y_pred

    def create_classification_report(self, 
        model, 
        X : list, 
        y : list, 
        output_dict : dict = True):
        """Returns a metrics classification report given the model and X and y sets.
        This shows the precision and recall of label predictions

        Args:
            model (object): the given model, LSTM in this case
            X (list): with training examples
            y (list): with training labels

        Returns:
            dataframe: with the metrics report to be printed
        """        
        y, y_pred = self.do_prediction(model, X, y)
        return classification_report(lstm_util.flatten_list(y), lstm_util.flatten_list(y_pred), 
            # labels=[0, 1, 2, 3, 4],
            # target_names=['long', 'short', 'elision', 'space', 'padding'],
            output_dict=output_dict, 
        )
    
    def create_X_y_sets(self, 
                        given_set: list, 
                        word2idx: list, 
                        label2idx: list, 
                        max_sentence_length: int):
        """Creates X and y sets that can be used by LSTM. Pads sequences and converts y to categorical

        Args:
            given_set (list): set to be converted to X and y
            word2idx (dict): with hashes for our syllables
            label2idx (dict): with hashes for our labels
            max_sentence_length (int): max length of a sentence: used for padding

        Returns:
            list: of X set
            list: of y set
        """        
        # now we map the sentences and labels to a sequence of numbers
        X = [[word2idx[w[0]] for w in s] for s in given_set]  # key 0 are labels
        y = [[label2idx[w[1]] for w in s] for s in given_set] # key 1 are labels
        # and then (post)pad the sequences using the PADDING label.
        X = tf.keras.utils.pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=word2idx[self.PADDING]) # value is our padding key
        y = tf.keras.utils.pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=label2idx[self.PADDING])
        # for training the network we also need to change the labels to categorial.
        y = np.array([tf.keras.utils.to_categorical(i, num_classes=len(self.LABELS)) for i in y])

        return X, y

    def kfold_model(sequence_labels: list, splits: int = 5):
        """
        """        
        # Convert the list of numpy arrays to a numpy array with numpy arrays
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)
        kf = KFold(n_splits=splits, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(sequence_labels):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, y_train = self.create_X_y_sets(
                sequence_labels_training_text, 
                self.word2idx, 
                self.label2idx, 
                self.max_sentence_length
            )

            model = self._get_model()

            model, history = self._fit_model(
                model = model,
                X = X_train,
                y = y_train,
                batch_size = 32,
                epochs = num_epochs,
                split = 0.2,
                verbose = True
            )   
            crf = self.fit_model(X_train, y_train)

            metrics_report = self.predict_model(crf, X_test, y_test)
            # TODO: just print a score for each run to terminal
            report_list.append(metrics_report)

        result = util.merge_kfold_reports(report_list)

        return result

    def load_model(self, model_name: str):
        return tf.keras.models.load_model(f"{conf.LSTM_MODELS_PATH}/{model_name}.keras")

    def _create_idx_dictionaries(self,
        unique_syllables: list, 
        labels: list):
        """This function creates the idx dictionaries needed for creating an LSTM on syllable hashes

        Args:
            unique_syllables (list): of unique syllables
            labels (list): of unique syllables

        Returns:
            dict: of syllable -> hash and label -> hash
        """        
        word2idx = {w: i for i, w in enumerate(unique_syllables)}
        label2idx = {t: i for i, t in enumerate(labels)}   
        return word2idx, label2idx     

    def _fit_model(self, 
        model,
        X: list,
        y: list,
        batch_size: int,
        epochs: int,
        split: float,
        verbose: bool
        ):

        history = model.fit(
            X, 
            y, 
            batch_size = batch_size, 
            epochs = epochs, 
            validation_split = split, 
            verbose = verbose
        )
        
        return model, history


    def _get_model(self):
        """Returns the LSTM model from the given parameters. Also allows a model to be loaded from disk

        Args:

        Returns:
            _type_: _description_
        """
        
        do_crf: bool = False

        embedding_output_dim: int = 25 # 25      
        lstm_layer_units: int = 10 # 10
        LEARNING_RATE: float = 0.001
            
        # Initiate the model structure
        input_layer = tf.keras.layers.Input(shape=(self.max_sentence_length,))
        
        model = tf.keras.layers.Embedding(
            input_dim = len(self.unique_syllables), 
            output_dim = embedding_output_dim, 
            input_length = self.max_sentence_length
        )(input_layer)

        # model = Dropout(0.1)(model)

        # model = tf.keras.layers.LSTM(
        #         units = lstm_layer_units, 
        #         return_sequences = True, 
        #         recurrent_dropout = 0.1
        #     )(model)

        model = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units = lstm_layer_units, 
                return_sequences = True, 
                recurrent_dropout = 0.1
            )
        )(model)

        model = tf.keras.layers.Dense(len(self.LABELS), activation='softmax')(model)

        # output_layer = TimeDistributed(Dense(50, activation="softmax"))(model)  # softmax output layer
        # kernel = TimeDistributed(Dense(num_labels, activation="softmax"))(model)  # softmax output layer

        if do_crf:
            crf = tfa.layers.CRF(units = len(self.LABELS))
            decoded_sequence, model, sequence_length, chain_kernel = crf(model)
            # tfa.losses.SigmoidFocalCrossEntropy(),
            # print(decoded_sequence.shape, potentials.shape, sequence_length.shape, chain_kernel.shape)

        model = tf.keras.models.Model(input_layer, model)

        model.compile(
            # optimizer="rmsprop",
            optimizer=tf.keras.optimizers.Adam(
                learning_rate = LEARNING_RATE
            ),  # Optimizer
            # Loss function to minimize
            loss=tf.keras.losses.CategoricalCrossentropy(),
            # loss=tfa.losses.SigmoidFocalCrossEntropy(),
            # List of metrics to monitor
            # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            # metrics=tf.keras.metrics.CategoricalAccuracy(),
            metrics=['accuracy'],
        )

        print(model.summary())

        return model        


    def _merge_sequence_label_files(self, file_paths):
        """Merges the given lists (contained in sequence labeling pickles) on the given path.
        Outputs one list with all sentences of the given texts in sequence labeling format.
        Useful when merging all metamorphoses for example.

        Args:
            texts (list): of sequence labeling pickled files
            path (string): where these pickled files are stored

        Returns:
            list: of merged texts
        """        
        line_list: list = []
        # Create a starting list from the last entry using pop
        for file in file_paths:
            lines = util.read_json(file)['lines']
            line_list += lines 

        return line_list
    
    def _retrieve_max_sentence_length(self, sequence_labels: list) -> int:
        """Returns the maximum sentence length of the given sequence label list. Used for padding calculations

        Args:
            sequence_labels (list): with sentences and their syllables and labels

        Returns:
            int: of maximum sentence length
        """        
        max_len : int = 0

        for sentence in sequence_labels:
            if len(sentence) > max_len:
                max_len = len(sentence)

        return max_len
