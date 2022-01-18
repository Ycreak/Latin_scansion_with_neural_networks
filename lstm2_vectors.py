import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

import utilities as util

# KERAS
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn_crfsuite import metrics as crf_metrics

# Based on https://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/

class LSTM_model():

    # Constants    
    PADDING = 'padding'

    num_epochs = 5
    load_model = False
    print_stats = False
    create_metric_reports = False

    TRAINING_TEXTS = ['VERG-aene.pickle'] #FIXME: cannot do multiple items
    TEST_TEXTS = ['VERG-aene.pickle']

    LABELS = np.array(['short', 'long', 'elision', PADDING])

    ZERO_VECTOR = np.zeros(util.cf.getint('Word2Vec', 'vector_size'))

    def __init__(self):       
        # Merge all training texts into one sequence label list
        sequence_labels_training_set = util.merge_sequence_label_lists(self.TRAINING_TEXTS, util.cf.get('Pickle', 'path_sequence_labels'))
        sequence_labels_test_set = util.merge_sequence_label_lists(self.TEST_TEXTS, util.cf.get('Pickle', 'path_sequence_labels'))

        # Get the information from this list we need to get the model working
        text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_training_set)
        max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_training_set)
        num_total_syllables = len(text_syllables)
        unique_syllables = list(set(text_syllables))
        # unique_syllables = np.append(unique_syllables, self.PADDING) # Append the padding key to the list of unique syllables
        num_unique_syllables = len(unique_syllables)
        num_labels = len(self.LABELS)

        word2vec_model = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'word2vec_model'))
        # print(word2vec_model)

        # vector = word2vec_model.wv['ar']
        # print(np.array(vector))

        # convert our unique syllables and lables into integers and store these in dictionaries
        word2idx = {w:  list(word2vec_model.wv[w]) for w in unique_syllables}
        word2idx['padding'] = self.ZERO_VECTOR # Append the padding vector to the list of unique syllables

        label2idx = {t: i for i, t in enumerate(self.LABELS)}

        # now we map the sentences and labels to a sequence of numbers
        X = [[word2idx[w[0]] for w in s] for s in sequence_labels_training_set]  # key 0 are labels
        y = [[label2idx[w[1]] for w in s] for s in sequence_labels_training_set] # key 1 are labels

        # max_padding_length = util.cf.getint('Word2Vec', 'vector_size') * max_sentence_length
        # and then (post)pad the sequences using the PADDING label.
        X = pad_sequences(maxlen=20, sequences=X, padding="post", value=self.ZERO_VECTOR, dtype='float32') # value is our padding key
        y = pad_sequences(maxlen=20, sequences=y, padding="post", value=label2idx[self.PADDING])

        # print(len(X[0]))
        # print(X[0])
        # exit(0)
        # for training the network we also need to change the labels to categorial.
        y = np.array([to_categorical(i, num_classes=num_labels) for i in y])

        # we split in train and test set.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = self.get_model( input_shape = max_sentence_length,
                                num_syllables = num_unique_syllables,
                                num_labels = num_labels,
                                X = X_train,
                                y = y_train,
                                epochs = self.num_epochs,
                                load_from_disk = self.load_model)
        
        # exit(0)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print('RESULT: loss -> {0}, accuracy -> {1}'.format(loss, accuracy))

        if self.create_metric_reports:
            print(self.create_metrics_report(model, X_test, y_test))

            confusion_matrix = self.create_confusion_matrix(model, X, y)
            df_confusion_matrix = pd.DataFrame(confusion_matrix, index = ['long', 'short', 'elision', 'padding'],
                                                columns = ['long', 'short', 'elision', 'padding'])
            # Drop the padding labels, as we don't need them (lstm scans them without confusion)
            df_confusion_matrix = df_confusion_matrix.drop('padding')
            df_confusion_matrix = df_confusion_matrix.drop(columns=['padding'])

            util.create_heatmap(dataframe = df_confusion_matrix,
                                ylabel =  'PREDICTED',
                                xlabel = 'TRUTH', 
                                title = 'CONFUSION MATRIX',
                                filename = 'confusion_matrix_lstm_aeneid',
                                vmax = 500
                                )
 
        #################
        # TEMP PRINTING #
        #################

        if self.print_stats:
            # Keep track of wrong scansions
            sentence_incorrect_counter = 0
            syllable_incorrect_counter = 0
            
            # Make a prediction over the whole dataset
            y_pred = model.predict(X)
            y_pred = np.argmax(y_pred, axis=-1)
            
            for idx, line in enumerate(X):
                # for every line in X, create a prediction and truth
                sentence = X[idx]          
                y_pred_current = y_pred[idx]
                y_true_current = np.argmax(y[idx], axis=-1)

                # If a line is not perfectly scanned, lets investigate
                if not (y_pred_current == y_true_current).all():
                    # count the syllables for every line that are not scanned correctly
                    syllable_incorrect_counter += (20 - np.count_nonzero(y_pred_current==y_true_current))
                    # retrieve the syllables from the hashes so we can read the sentence again 11
                    syllable_list = []
                    for item in sentence:
                        syllable_list.append(list(word2idx.keys())[list(word2idx.values()).index(item)])                  
                    # now, for each line, print some information to investigate the problems
                    while 'PADDING' in syllable_list: syllable_list.remove('PADDING')    
                    print('syllables : ', syllable_list)
                    # print('prediction: ', y_pred_current)
                    # print('truth:      ', y_true_current)
                    # also, for latinists, print the scansion
                    y_pred_labels = ['—' if j==0 else '⏑' if j==1 else 'e' if j==2 else j for j in y_pred_current]
                    y_train_labels = ['—' if j==0 else '⏑' if j==1 else 'e' if j==2 else j for j in y_true_current]

                    while 3 in y_pred_labels: y_pred_labels.remove(3)    
                    while 3 in y_train_labels: y_train_labels.remove(3)    

                    print('prediction: ', y_pred_labels)
                    print('truth     : ', y_train_labels)
                    
                    print('\n##########################\n')
                    # count the sentence as incorrect
                    sentence_incorrect_counter += 1
                    
            # after all scrutiny, print the final statistics                    
            score_sentences = round(sentence_incorrect_counter/len(X)*100,2)
            score_syllables = round(syllable_incorrect_counter/num_total_syllables*100,2)

            print('SENTENCES SCANNED WRONGLY: ', sentence_incorrect_counter)
            print('PERCENTAGE WRONG: {0}%'.format(score_sentences))
            
            print('SYLLABLES SCANNED WRONGLY: ', syllable_incorrect_counter)
            print('PERCENTAGE WRONG: {0}%'.format(score_syllables))

    def flatten_list(self, given_list):
        return [item for sublist in given_list for item in sublist]


    def create_confusion_matrix(self, model, X, y):
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
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=-1)
        y = np.argmax(y, axis=-1)

        flat_list = [item for sublist in y for item in sublist]
        flat_list2 = [item for sublist in y_pred for item in sublist]

        return confusion_matrix(flat_list, flat_list2)

    def create_metrics_report(self, model, X, y):
        """Returns a metrics classification report given the model and X and y sets.
        This shows the precision and recall of label predictions

        Args:
            model (object): the given model, LSTM in this case
            X (list): with training examples
            y (list): with training labels

        Returns:
            dataframe: with the metrics report to be printed
        """        
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=-1)
        y = np.argmax(y, axis=-1)

        metrics_report = crf_metrics.flat_classification_report(
            y, y_pred, labels=[0,1,2], target_names=['long', 'short', 'elision'], digits=4
        )

        return metrics_report

    def get_model(self, input_shape, num_syllables, num_labels, X, y, epochs, load_from_disk):
        """Returns the LSTM model from the given parameters. Also allows a model to be loaded from disk

        Args:
            max_len (int): max sentence length
            num_syllables (int): number of unique syllables
            num_labels (int): number of labels to predict
            X (list): with training examples
            y (list): with training labels
            epochs (int): number of epochs to train the model
            load_from_disk (bool): whether to load the model from disk

        Returns:
            object: lstm model
        """
        if load_from_disk:
            return keras.models.load_model('./model')


        # print(input_shape, num_syllables, num_labels, X[0], y[0])

        # Initiate the model structure
        input = Input(shape=(input_shape,))
        
        model = Embedding(input_dim=num_syllables, output_dim=50, input_length=input_shape)(input)  # 50-dim embedding
        model = Dropout(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)  # variational biLSTM
        
        out = TimeDistributed(Dense(num_labels, activation="softmax"))(model)  # softmax output layer

        model = Model(input, out)

        # exit(0)
        # Compile the model
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(X, y, batch_size=32, epochs=epochs, verbose=1)
        
        # model.save('./model')

        util.create_line_plot(
            plots = (history.history['accuracy'],),
            ylabel = 'Accuracy',
            xlabel = 'Epoch',
            plot_titles = ['train'],
            title = 'LSTM accuracy',
            plotname = 'lstm_accuracy'
        )

        return model

    def retrieve_max_sentence_length(self, sequence_labels):
        """Returns the maximum sentence length of the given sequence label list. Used for padding calculations

        Args:
            sequence_labels (list): with sentences and their syllables and labels

        Returns:
            int: of maximum sentence length
        """        
        max_len = 0
        for sentence in sequence_labels:
            if len(sentence) > max_len:
                max_len = len(sentence)
        return max_len

    def retrieve_syllables_from_sequence_label_list(self, sequence_labels):
        """returns all the syllables from the given sequence label list 

        Args:
            sequence_labels (list): with sequence labels

        Returns:
            list: of all syllables in the given texts
        """
        unique_syllable_list = []

        for sentence in sequence_labels:
            for syllable, label in sentence:
                unique_syllable_list.append(syllable)

        return unique_syllable_list

if __name__ == "__main__":
    lstm = LSTM_model()