from gettext import textdomain
from locale import D_FMT
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

import utilities as util
from datetime import datetime
import random

import argparse

# LSTM related imports
from tensorflow import keras
# from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from keras.models import Model #, Input

from keras.layers import Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn_crfsuite import metrics as crf_metrics

# Based on https://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/

class LSTM_model():

    # Constants    
    PADDING = 'padding'

    # num_epochs = 25
    print_stats = True
    metric_report = True
    confusion_matrix = True

    evaluate = False

    TRAINING_TEXTS = ['OV-amo1.pickle']
    TEST_TEXTS = ['OV-amo1.pickle']
    ALL_TEXTS = TRAINING_TEXTS + TEST_TEXTS

    LABELS = np.array(['long', 'short', 'elision', 'space', PADDING])

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        # Debugging
        # FLAGS.anceps = True
        # FLAGS.single_text = True
        # FLAGS.create_model = False
        # FLAGS.save_model = True
        # FLAGS.epochs = 1
        # FLAGS.metrics_report = True
        # FLAGS.verbose = True
        
        
        self.num_epochs = FLAGS.epochs
        self.split_size = FLAGS.split

        if FLAGS.anceps:
            self.LABELS = np.array(['long', 'short', 'elision', 'space', self.PADDING, 'anceps'])

        # FLAGS.single_text = True
        # self.num_epochs = 1

        # self.print_length_sequence_label_files()
        # exit(0)

        # To make the LSTM with integer hashing working, we need to make a list of all syllables from all the texts we are looking at 
        # First, find all our pickle files   
        all_sequence_label_pickles = util.Create_files_list(util.cf.get('Pickle', 'path_sequence_labels'), 'pickle') 
        # Merge them into one big file list
        sequence_labels_all_set = util.merge_sequence_label_lists(all_sequence_label_pickles, util.cf.get('Pickle', 'path_sequence_labels')) 
        # Retrieve all syllables from this big list
        all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
        # We need to extract the max sentence length over all these texts to get the padding correct later
        max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)       
        # And we need to create a list of all unique syllables for our word2idx one-hot encoding
        unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING)
        word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)            

        # With that out of the way, we can start the LSTM process
        print('\nPreprocessing done:')
        print('number of syllables: ', len(unique_syllables))
        print('number of labels: ', len(self.LABELS))
        print('max_sentence_length: ', max_sentence_length)
        print('\n')

        if FLAGS.single_text:
            text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'),'HEX_ELE-all.pickle')
            self.run_idx_lstm_single_text(text, 
                                          do_evaluate=True, 
                                          do_metric_report=True, 
                                          do_confusion_matrix=True, 
                                          print_stats=False)

        if FLAGS.kfold:
            text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), FLAGS.train)
            
            all_text_syllables = self.retrieve_syllables_from_sequence_label_list(text)
            max_sentence_length = self.retrieve_max_sentence_length(text)
            unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING) # needs to be sorted for word2idx consistency!            
            word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)

            X, y = self.create_X_y_sets(text, word2idx, label2idx, max_sentence_length)

            # Perform kfold to check if we don't have any overfitting
            result = self.kfold_model(text, X, y, 5, max_sentence_length, unique_syllables)
            print(result)

        # if True:
        #     train_texts = ['SEN-proofread.pickle']
        #     test_texts = ['SEN-aga.pickle']
        #     self.do_experiment(train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='seneca_anceps', plot_title='Trimeter Texts')


        if FLAGS.custom_train_test:
            
            train_texts = ['HEX_ELE-all.pickle'] #['VERG-aene.pickle']#
            test_texts = ['anapest.pickle', 'dimeter.pickle', 'hendycasyllable.pickle', 'glyconee.pickle', 'tetrameter.pickle', 'trimeter.pickle'] #FLAGS.test
            self.do_experiment(train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='seneca_precise', plot_title='Trimeter Texts')

            # train_texts = ['SEN-proofread.pickle']
            # test_texts = ['SEN-aga.pickle']
            # self.do_experiment(train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='seneca_precise', plot_title='Trimeter Texts')





    def kfold_model(self, sequence_labels, X, y, splits, max_sentence_length, unique_syllables):
        """Performs a kfold cross validation of a LSTM model fitted and trained on the given data

        Args:
            sequence_labels (list): of sequence labels and their features
            X (numpy array): of training/test examples
            y (numpy array): of labels
            splits (int): of number of splits required

        Returns:
            dict: with results of the cross validation
        """        
        if util.cf.get('Util', 'verbose'): print('Predicting the model')

        report_list = []

        # Convert the list of numpy arrays to a numpy array with numpy arrays
        # X = np.array(X, dtype=object)
        # y = np.array(y, dtype=object)
        kf = KFold(n_splits=splits, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(sequence_labels):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = self.get_model( max_len = max_sentence_length,
                                    num_syllables = len(unique_syllables),
                                    num_labels = len(self.LABELS),
                                    X = X_train,
                                    y = y_train,
                                    epochs = FLAGS.epochs,
                                    create_model = True,
                                    save_model = FLAGS.save_model)

            metrics_report = self.create_metrics_report(model, X_test, y_test, output_dict=True)
            # TODO: just print a score for each run to terminal
            report_list.append(metrics_report)

        result = util.merge_kfold_reports(report_list)

        return result

    def combine_sequence_label_lists(self, key, name):
        # TO COMBINE LISTS
        name = name + '.pickle'
        my_list = sorted(util.Create_files_list(util.cf.get('Pickle', 'path_sequence_labels'), key))
        temp = util.merge_sequence_label_lists(my_list, util.cf.get('Pickle', 'path_sequence_labels'))
        util.Pickle_write(util.cf.get('Pickle', 'path_sequence_labels'), name, temp)

    def print_length_sequence_label_files(self):
        # TO FIND LENGTH
        my_list = sorted(util.Create_files_list  (util.cf.get('Pickle', 'path_sequence_labels'), '.pickle'))
        for text in my_list:
            current_text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), text)
            print(text, len(current_text))


    def do_experiment(self, train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='default', plot_title='default'):
        # import matplotlib.pyplot as plt        
        
        column_names = ["predictor", "predictee", "score"]
        
        # df = pd.DataFrame(columns = [])

        df = pd.DataFrame(columns = ["epochs", "anapest_long", "anapest_short",
                                        'dimeter_long', 'dimeter_short',
                                        'tetrameter_long','tetrameter_short',
                                        'glyconee_long', 'glyconee_short',
                                        'hendycasyllable_long', 'hendycasyllable_short',
                                        'trimeter_long', 'trimeter_short'])



        # df.to_csv('epoch_learning_rate_testing.csv', mode='w', index=False, header=True)

        new_line = {'epochs': self.num_epochs}
        df = df.append(new_line, ignore_index=True)    

        df_long = pd.DataFrame(columns = column_names)
        df_short = pd.DataFrame(columns = column_names)
        df_elision = pd.DataFrame(columns = column_names)

        # Merge the training and test texts and retrieve from it all data we need to get the model working
        # all_texts = train_texts + test_texts
        # sequence_labels_all_set = util.merge_sequence_label_lists(all_texts, util.cf.get('Pickle', 'path_sequence_labels'))

        # all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
        # max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)
        # unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING)
        # word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)

        for train_text in train_texts:


            sequence_labels_training_text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), train_text)
            X_train, y_train = self.create_X_y_sets(sequence_labels_training_text, word2idx, label2idx, max_sentence_length)

            model = self.get_model( max_len = max_sentence_length,
                                    num_syllables = len(unique_syllables),
                                    num_labels = len(self.LABELS),
                                    X = X_train,
                                    y = y_train,
                                    epochs = FLAGS.epochs,
                                    create_model = FLAGS.create_model,
                                    save_model = FLAGS.save_model,
                                    model_name = train_text.split('.')[0])

            for test_text in test_texts:

                sequence_labels_test_text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), test_text)
                X_test, y_test = self.create_X_y_sets(sequence_labels_test_text, word2idx, label2idx, max_sentence_length)
                classification_report = self.create_classification_report(model, X_test, y_test)

                # print(test_text, classification_report)

                predictee = test_text.split('.')[0]#.capitalize()
                score_long = float(round(classification_report['0']['f1-score'],2))
                score_short = float(round(classification_report['1']['f1-score'],2))
                # score_elision = float(round(classification_report['2']['f1-score'],2))
                
                # print(score_long, score_short, predictee)
                # new_line_long = {'epochs': self.num_epochs, 'predictee': predictee+'_long', 'long': score_long, 'short': score_short}
                # new_line_short = {'epochs': self.num_epochs, 'predictee': predictee+'_short', 'short': score_short}

                predictee_long = predictee+'_long'
                predictee_short = predictee+'_short'

                df.loc[df['epochs'] == self.num_epochs, [predictee_long]] = score_long
                df.loc[df['epochs'] == self.num_epochs, [predictee_short]] = score_short


                # df = df.append(new_line, ignore_index=True)    

                print(df)

                continue
                # exit(0)

                # if self.FLAGS.metrics_report:
                #     print_metrics_report = self.create_confusion_matrix(model, X_test, y_test)
                #     # self.create_seaborn_heatmap(print_metrics_report)
                    
                    
                #     print(print_metrics_report)
                #     print_metrics_report = self.create_metrics_report(model, X_test, y_test, output_dict=True)
                #     print(print_metrics_report)                    
                    
                    # exit(0)

                predictor = train_text.split('-')[0].capitalize() # get names
                predictee = test_text.split('-')[0].capitalize()

                score_long = float(round(metrics_report['long']['f1-score'],4)) # save the score
                score_short = float(round(metrics_report['short']['f1-score'],4)) # save the score
                score_elision = float(round(metrics_report['elision']['f1-score'],4)) # save the score
                
                # Add score to the dataframe for our heatmap
                new_line_long = {'predictor': predictor, 'predictee': predictee, 'score': score_long}
                new_line_short = {'predictor': predictor, 'predictee': predictee, 'score': score_short}
                new_line_elision = {'predictor': predictor, 'predictee': predictee, 'score': score_elision}

                df_long = df_long.append(new_line_long, ignore_index=True)    
                df_short = df_short.append(new_line_short, ignore_index=True)    
                df_elision = df_elision.append(new_line_elision, ignore_index=True)   


        # df.to_csv('epoch_learning_rate_testing.csv', mode='a', index=False, header=False)
        df.to_csv('epoch_learning_rate_testing2.csv', mode='w', index=False, header=True)

        print(df)
        exit(0)

        time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        heatmap_data = pd.pivot_table(df_long, values='score', index=['predictor'], columns='predictee')
        heatmap_data.to_csv('./csv/{0}_f1-scores_long.csv'.format(time))
        myplot = plt.figure(figsize=(16,10))
        myplot = plt.subplot(2, 2, 1)
        myplot = util.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'test',
                        ylabel = 'train',
                        title = '{0}: Long f1-scores'.format(plot_title),
                        filename = '{0}-long'.format(exp_name),
                        path = './plots/experiments/')

        heatmap_data = pd.pivot_table(df_short, values='score', index=['predictor'], columns='predictee')
        heatmap_data.to_csv('./csv/{0}_f1-scores_short.csv'.format(time))
        myplot = plt.subplot(2, 2, 2)        
        myplot = util.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'test',
                        ylabel = 'train',
                        title = '{0}: Short f1-scores'.format(plot_title),
                        filename = '{0}-short'.format(exp_name),
                        path = './plots/experiments/')

        heatmap_data = pd.pivot_table(df_elision, values='score', index=['predictor'], columns='predictee')
        heatmap_data.to_csv('./csv/{0}_f1-scores_elision.csv'.format(time))
        myplot = plt.subplot(2, 2, 3)
        myplot = util.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'test',
                        ylabel = 'train',
                        title = '{0}: Elision f1-scores'.format(plot_title),
                        filename = '{0}-elision'.format(exp_name),
                        path = './plots/experiments/')

    def create_idx_dictionaries(self, unique_syllables, labels):
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

    def create_X_y_sets(self, given_set, word2idx, label2idx, max_sentence_length):
        """Creates X and y sets that can be used by LSTM. Pads sequences and converts y to categorical

        Args:
            given_set (list): set to be converted to X and y
            word2idx (dict): with hashes for our syllables
            label2idx (dict): with hashes for our labels
            max_sentence_length (int): max length of a sentence: used for padding

        Returns:
            list: of X and y sets
        """        
        # now we map the sentences and labels to a sequence of numbers
        X = [[word2idx[w[0]] for w in s] for s in given_set]  # key 0 are labels
        y = [[label2idx[w[1]] for w in s] for s in given_set] # key 1 are labels
        # and then (post)pad the sequences using the PADDING label.
        X = tf.keras.utils.pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=word2idx[self.PADDING]) # value is our padding key
        y = tf.keras.utils.pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=label2idx[self.PADDING])
        # for training the network we also need to change the labels to categorial.
        y = np.array([to_categorical(i, num_classes=len(self.LABELS)) for i in y])

        return X, y

    def do_prediction(self, model, X, y):
        """Does a prediction given the model, X and y sets

        Args:
            model (_type_): _description_
            X (_type_): _description_
            y (_type_): _description_
        """        
        y_pred = model.predict(X)

        # ['long', 'short', 'elision', 'space', self.PADDING, 'anceps']


        if self.FLAGS.anceps: # We dont want to predict the label anceps, so we delete it from the possible predictions
            for line in y_pred:
                for syllable in line:
                    # print(syllable)
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
                    # else:
                        

            # print(y_pred[0]) # = y_pred[:, :, :-1]
            
        y_pred = np.argmax(y_pred, axis=-1)
        y = np.argmax(y, axis=-1)

        return y, y_pred

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
        # y_pred = model.predict(X)

        # if self.FLAGS.anceps: # We dont want to predict the label anceps, so we delete it from the possible predictions
        #     y_pred = y_pred[:, :, :-1]

        # y_pred = np.argmax(y_pred, axis=-1)
        # y = np.argmax(y, axis=-1)
        y, y_pred = self.do_prediction(model, X, y)

        flat_list = [item for sublist in y for item in sublist]
        flat_list2 = [item for sublist in y_pred for item in sublist]

        return confusion_matrix(flat_list, flat_list2)

    def flatten_list(self, given_list: list) -> list:
        return [item for sublist in given_list for item in sublist]

    def create_classification_report(self, model, X, y, output_dict=True):
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
        return classification_report(self.flatten_list(y), self.flatten_list(y_pred), 
            # labels=[0, 1, 2, 3, 4],
            # target_names=['long', 'short', 'elision', 'space', 'padding'],
            output_dict=True, 
        )

    def evaluate_model(self, model, X, y):
        """Evaluates the given model on the given X and y sets

        Args:
            model ([type]): [description]
            X ([type]): [description]
            y ([type]): [description]

        Returns:
            loss, accuracy: [description]
        """        
        loss, accuracy = model.evaluate(X, y, verbose=self.FLAGS.verbose)
        return loss, accuracy

    def get_model(self, max_len, num_syllables, num_labels, X, y, epochs, create_model, save_model, model_name='default'):
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
        path = './models/lstm/' + model_name    # Get model path name for saving or loading
        
        do_crf: bool = False

        import tensorflow as tf
        import tensorflow_addons as tfa
        from tensorflow_addons.layers import CRF

        embedding_output_dim: int = 25 # 25      
        lstm_layer_units: int = 10 # 10
        BATCH_SIZE: int = 32 # 64
        EPOCHS: int = self.num_epochs
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

            model = tf.keras.layers.LSTM(
                    units = lstm_layer_units, 
                    return_sequences = True, 
                    recurrent_dropout = 0.1
                )(model)

            # model = tf.keras.layers.Bidirectional(
            #     tf.keras.layers.LSTM(
            #         units = lstm_layer_units, 
            #         return_sequences = True, 
            #         recurrent_dropout = 0.1
            #     )
            # )(model)

            model = tf.keras.layers.Dense(num_labels, activation='softmax')(model)

            # output_layer = TimeDistributed(Dense(50, activation="softmax"))(model)  # softmax output layer
            # kernel = TimeDistributed(Dense(num_labels, activation="softmax"))(model)  # softmax output layer

            if do_crf:
                crf = CRF(units = num_labels)
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
                verbose = self.FLAGS.verbose
            )
            
            if save_model: 
                model.save(path)

            return model        
        
        else:
            return tf.keras.models.load_model(path)



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
        syllable_list = []

        for sentence in sequence_labels:
            for syllable, label in sentence:
                syllable_list.append(syllable)

        return syllable_list

    def create_seaborn_heatmap(self, confusion_matrix):
        if self.FLAGS.anceps:
            labels = ['long', 'short', 'elision', 'space', 'padding', 'anceps']
        else:
            labels = ['long', 'short', 'elision', 'space', 'padding']

        df_confusion_matrix = pd.DataFrame(confusion_matrix, index = labels,
                                        columns = labels)
        # Drop the padding labels, as we don't need them (lstm scans them without confusion): delete both row and column
        df_confusion_matrix = df_confusion_matrix.drop('padding')
        df_confusion_matrix = df_confusion_matrix.drop(columns=['padding', 'space'])

        df_confusion_matrix = df_confusion_matrix.drop('space')
        
        if self.FLAGS.anceps:
            df_confusion_matrix = df_confusion_matrix.drop(columns=['anceps'])
            df_confusion_matrix = df_confusion_matrix.drop('anceps')            
        
        # df_confusion_matrix = df_confusion_matrix.drop(columns=['padding'])

        util.create_heatmap(dataframe = df_confusion_matrix,
                            xlabel = 'TRUTH', 
                            ylabel =  'PREDICTED',
                            title = 'CONFUSION MATRIX',
                            filename = 'new_confusion_matrix',
                            # vmax = 500
                            )         

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # p.add_argument("--create_model", action="store_true", help="specify whether to create the model: if not specified, we load from disk")
    # p.add_argument("--save_model", action="store_true", help="specify whether to save the model: if not specified, we do not save")
    p.add_argument("--single_text", action="store_true", help="specify whether to run the single text LSTM function")
    # p.add_argument("--exp_hexameter", action="store_true", help="specify whether to run the hexameter LSTM experiment")
    # p.add_argument("--exp_transfer", action="store_true", help="specify whether to run the hexameter transerability LSTM experiment")
    # p.add_argument("--exp_elegiac", action="store_true", help="specify whether to run the hexameter genre LSTM experiment")
    # p.add_argument("--exp_train_test", action="store_true", help="specify whether to run the train/test split LSTM experiment")
    # p.add_argument("--exp_transfer_boeth", action="store_true", help="specify whether to run the Boeth LSTM experiment")
    p.add_argument("--model_predict", action="store_true", help="let a specific model predict a specific text")
    p.add_argument("--metrics_report", action="store_true", help="specifiy whether to print the metrics report")
    # p.add_argument("--kfold", action="store_true", help="specifiy whether to run the kfold experiment")

    p.add_argument("--anceps", action="store_true", help="specify whether to train on an anceps text")
    # p.add_argument("--verbose", action="store_true", help="specify whether to run the code in verbose mode")
    
    # p.add_argument('--epochs', default=25, type=int, help='number of epochs')
    p.add_argument("--split", type=util.restricted_float, default=0.2, help="specify the split size of train/test sets")

    p.add_argument("--create_model", action="store_true", 
                    help="specify whether to create the model: if not specified, we load from disk")
    p.add_argument("--save_model", action="store_true", 
                    help="specify whether to save the model: if not specified, we do not save")
    p.add_argument("--verbose", action="store_true", 
                    help="specify whether to run the code in verbose mode")
    p.add_argument('--epochs', default=25, type=int, 
                    help='number of epochs')

    p.add_argument('--custom_train_test', action="store_true", 
                    help='Run the model with the provided train and test set')
    p.add_argument('--kfold', action="store_true", 
                    help='Train and test the model on the same training set using kfold cross validation')
    p.add_argument('--train', default='none', type=str, 
                    help='Dataset to train the model on')
    p.add_argument('--test', default='none', type=str, 
                    help='Dataset to test the model on')


    FLAGS = p.parse_args()    
    
    lstm = LSTM_model(FLAGS)
