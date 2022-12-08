# Standard library imports

# Third party library imports
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sklearn.metrics

# Local imports
import utilities as util
import config as conf

# Based on https://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/

class Latin_LSTM():
    """This class provides the user with tools to create an LSTM model for scanning Latin. On startup,
    it prepares the padding procedures and creates syllable and label dictionaries for the one-hot encoding
    used by the LSTM. Having done that, the class can create and save models and predict custom syllable-label lists.
    """    
    # class constants    
    PADDING = 'padding'
    LABELS = np.array(['long', 'short', 'elision', 'space', PADDING])

    def __init__(
        self,
        sequence_labels_folder: str,
        models_save_folder: str,
        anceps_label: bool = False):

        self.anceps_label = anceps_label
        self.sequence_labels_folder = sequence_labels_folder
        self.models_save_folder = models_save_folder

        if self.anceps_label:
            self.LABELS = np.array(['long', 'short', 'elision', 'space', self.PADDING, 'anceps'])

        # To make the LSTM with integer hashing working, we need to make a list of all syllables from all the texts we are looking at 
        # First, find all our pickle files   
        all_sequence_label_pickles = util.create_files_list(self.sequence_labels_folder, 'pickle') 
        # Merge them into one big file list
        sequence_labels_all_set = util.merge_sequence_label_lists(all_sequence_label_pickles, self.sequence_labels_folder) 
        # Retrieve all syllables from this big list
        all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
        # We need to extract the max sentence length over all these texts to get the padding correct later
        self.max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)       
        # And we need to create a list of all unique syllables for our word2idx one-hot encoding
        self.unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING)
        self.word2idx, self.label2idx = self.create_idx_dictionaries(self.unique_syllables, self.LABELS)            

        # With that out of the way, we can start the LSTM process
        print('Preprocessing done.')
        print('number of syllables: ', len(self.unique_syllables))
        print('number of labels: ', len(self.LABELS))
        print('max_sentence_length: ', self.max_sentence_length)

    def predict_given_set(
        self, 
        given_set: list,
        model) -> list:
        """Given the set and the model, this function returns a list with predictions.

        Args:
            given_set (list): with sequences to be predicted
            model (tensorflow model): that has to predict the given set 

        Returns:
            list: with predictions per given syllable in the given_set
        """        
        # now we map the sentences and labels to a sequence of numbers
        X = [[self.word2idx[w] for w in s] for s in given_set]  # key 0 are labels
        # and then (post)pad the sequences using the PADDING label.
        X = tf.keras.utils.pad_sequences(
                maxlen = self.max_sentence_length, 
                sequences = X, 
                padding = "post", 
                value = self.word2idx[self.PADDING]
                ) # value is our padding key

        # model = tf.keras.models.load_model(self.lstm_model_path)

        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=-1)

        return y_pred

    def create_idx_dictionaries(self,
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

    def do_prediction(self, 
        model, 
        X: list, 
        y: list):
        """Does a prediction given the model, X and y sets

        Args:
            model (tensorflow model): that needs to do the predicting
            X (list): with syllables that need predicting
            y (_type_): with labels that are the predictions
        """        
        y_pred = model.predict(X)

        if self.anceps_label: # We dont want to predict the label anceps, so we delete it from the possible predictions
            for line in y_pred:
                for syllable in line:
                    # check which label has the highest confidence
                    position = np.where(syllable == np.amax(syllable))
                    # if it is the anceps label, check confidence for long and short
                    if position[0][0] == 5:
                        long = syllable[0]
                        short = syllable[1]
                        if long > short:
                            syllable = np.array([1, 0, 0, 0, 0, 0])
                        else:
                            syllable = np.array([0, 1, 0, 0, 0, 0])                        
        # generate y_pred and return per syllable the integer (syllable) with the highest confidence            
        y_pred = np.argmax(y_pred, axis=-1)
        y = np.argmax(y, axis=-1)

        return y, y_pred

    def create_model(self, 
        num_epochs : int,
        text : str, 
        save_model : bool = True,
        model_name : str = 'default'):
        """Function to create a model given a text (pickled sequence label file). Returns the
        created model. In addition, allows the model to be saved to disk.

        Args:
            text (string): name of pickled sequence label file to be used for training
            save_model (bool, optional): whether the created model is to be saved to disk. Defaults to True.
            model_name (str, optional): name to be given to the saved model (folder name). Defaults to 'default'.

        Returns:
            tensorflow model: containing the trained LSTM
        """        
        sequence_labels_training_text = util.pickle_read(self.sequence_labels_folder, text)
        X_train, y_train = self.create_X_y_sets(sequence_labels_training_text, 
                                                self.word2idx, 
                                                self.label2idx, 
                                                self.max_sentence_length)

        model = self.get_model(
            max_len = self.max_sentence_length,
            num_syllables = len(self.unique_syllables),
            num_labels = len(self.LABELS),
            X = X_train,
            y = y_train,
            epochs = num_epochs,
            create_model = True,
            save_model = save_model,
            model_name = model_name
        )   

        return model

    def create_confusion_matrix(self, 
        model, 
        X : list, 
        y : list):
        """Creates a confusion matrix from the given model, X and y sets. As y is one-hot encoded, we need to take
        the argmax value. Secondly, because the list of lists structure (sentences), we need to flatten both prediction
        lists in order to pass them to the confusion_matrix function.

        Args:
            model (object): of lstm model
            X (list): list of lists with sentences encoded as integers
            y (list): of labels, same as X.

        Returns:
            confusion matrix: of labels
        """        
        if self.FLAGS.anceps: # We dont want to predict the label anceps, so we delete it from the possible predictions
            y_pred = y_pred[:, :, :-1]

        y, y_pred = self.do_prediction(model, X, y)

        return sklearn.metrics.confusion_matrix(self.flatten_list(y), self.flatten_list(y_pred))

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
        return sklearn.metrics.classification_report(self.flatten_list(y), self.flatten_list(y_pred), 
            # labels=[0, 1, 2, 3, 4],
            # target_names=['long', 'short', 'elision', 'space', 'padding'],
            output_dict=True, 
        )

    def flatten_list(self, given_list: list) -> list:
        """Flattens the given list. Meaning that nested lists inside the given lists are
        turned into one list. For example, [[a,b],[c]] -> [a,b,c]

        Args:
            given_list (list): nested list that needs to be flattened

        Returns:
            list: flattened list
        """        
        return [item for sublist in given_list for item in sublist]

    def get_model(self, 
        max_len : int, 
        num_syllables : int, 
        num_labels : int, 
        X : list, 
        y : list, 
        epochs : int, 
        create_model : bool, 
        save_model : bool, 
        model_name : str = 'default'):
        """Returns the LSTM model from the given parameters. Also allows a model to be loaded from disk

        Args:
            max_len (int): max sentence length
            num_syllables (int): number of unique syllables
            num_labels (int): number of labels to predict
            X (list): with training examples
            y (list): with training labels
            epochs (int): number of epochs to train the model
            create_model (bool): whether to create the model or load it from disk
            save_model (bool): whether to save the model to disk
            model_name (str, optional): name of saved folder in case of saving. Defaults to 'default'.

        Returns:
            _type_: _description_
        """
        path = self.models_save_folder + model_name # get model path name for saving or loading
        
        do_crf: bool = False

        embedding_output_dim: int = 25 # 25      
        lstm_layer_units: int = 10 # 10
        BATCH_SIZE: int = 32 # 64
        EPOCHS: int = epochs
        LEARNING_RATE: float = 0.001

        if create_model:
            
            # Initiate the model structure
            input_layer = tf.keras.layers.Input(shape=(max_len,))
            
            model = tf.keras.layers.Embedding(
                input_dim = num_syllables, 
                output_dim = embedding_output_dim, 
                input_length = max_len
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

            model = tf.keras.layers.Dense(num_labels, activation='softmax')(model)

            # output_layer = TimeDistributed(Dense(50, activation="softmax"))(model)  # softmax output layer
            # kernel = TimeDistributed(Dense(num_labels, activation="softmax"))(model)  # softmax output layer

            if do_crf:
                crf = tfa.layers.CRF(units = num_labels)
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

            history = model.fit(X, y, 
                batch_size = BATCH_SIZE, 
                epochs = epochs, 
                validation_split = 0.1, 
                verbose = True
            )
            
            if save_model: 
                model.save(path)

            return model        
        
        else:
            return tf.keras.models.load_model(path)

    def retrieve_max_sentence_length(self, sequence_labels: list) -> int:
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

    def retrieve_syllables_from_sequence_label_list(self, sequence_labels: list) -> list:
        """returns all the syllables from the given sequence label list 

        Args:
            sequence_labels (list): with sequence labels

        Returns:
            list: of all syllables in the given texts
        """
        syllable_list = []

        for sentence in sequence_labels:
            for syllable, label in sentence:
                syllable_list.append(syllable)

        return syllable_list

if __name__ == "__main__":

    lstm = Latin_LSTM(
        sequence_labels_folder = conf.SEQUENCE_LABELS_FOLDER,
        models_save_folder = './models/lstm/',
        anceps_label = False,
        ) 

    model = lstm.create_model(
        num_epochs = 2,
        text = 'HEX_ELE-all.pickle', 
        save_model = True, 
        model_name = 'temp'
        )

        
