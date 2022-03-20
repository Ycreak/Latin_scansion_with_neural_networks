from gettext import textdomain
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

import utilities as util
from datetime import datetime
import random

import argparse
# KERAS
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Model, Input
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

    TRAINING_TEXTS = ['VERG-aene.pickle']
    TEST_TEXTS = ['OV-amo1.pickle']
    ALL_TEXTS = TRAINING_TEXTS + TEST_TEXTS

    LABELS = np.array(['long', 'short', 'elision', 'space', PADDING])

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        # Debugging
        # FLAGS.anceps = True
        # FLAGS.single_text = True
        # FLAGS.create_model = True
        # FLAGS.epochs = 1
        
        
        self.num_epochs = FLAGS.epochs
        self.split_size = FLAGS.split

        if FLAGS.anceps:
            self.LABELS = np.array(['long', 'short', 'elision', 'space', self.PADDING, 'anceps'])

        # FLAGS.single_text = True
        # self.num_epochs = 1

        # self.print_length_sequence_label_files()
        # exit(0)

        # To make the LSTM with integer hashing working, we need to make a list of all syllables from all the texts we are looking at    
        all_sequence_label_pickles = util.Create_files_list(util.cf.get('Pickle', 'path_sequence_labels'), 'pickle') # Find all pickle files
        sequence_labels_all_set = util.merge_sequence_label_lists(all_sequence_label_pickles, util.cf.get('Pickle', 'path_sequence_labels')) # Merge them into one big file list
        all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
        # We need to extract the max sentence length over all these texts to get the padding correct later
        max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)       
        # And we need to create a list of all unique syllables for our word2idx one-hot encoding
        unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING)
        word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)            

        # With that out of the way, we can start the LSTM process
        print('done')
        # Quickly create models
        # train_texts = ['VERG-aene.pickle', 'HEX-all.pickle', 'ELE-all.pickle', 'HEX_ELE-all.pickle']
        # test_texts = ['PERS-satu.pickle']
        # all_texts = train_texts + test_texts
        # sequence_labels_all_set = util.merge_sequence_label_lists(all_texts, util.cf.get('Pickle', 'path_sequence_labels'))

        # self.do_experiment(train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='training', plot_title='Training Texts')

        # exit(0)
        # all_text_syllables = self.retrieve_syllables_from_sequence_label_list(util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), 'VERG-aene.pickle'))
        # max_sentence_length = self.retrieve_max_sentence_length(util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), 'VERG-aene.pickle'))       
        # unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING)
        # word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)

        # exit(0)

        if FLAGS.single_text:
            text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'),'HEX_ELE-all.pickle')
            self.run_idx_lstm_single_text(text, 
                                          do_evaluate=True, 
                                          do_metric_report=True, 
                                          do_confusion_matrix=True, 
                                          print_stats=False)

        if FLAGS.kfold:
            text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), 'OV-ele.pickle')
            
            all_text_syllables = self.retrieve_syllables_from_sequence_label_list(text)
            max_sentence_length = self.retrieve_max_sentence_length(text)
            unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING) # needs to be sorted for word2idx consistency!            
            word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)

            X, y = self.create_X_y_sets(text, word2idx, label2idx, max_sentence_length)

            # Perform kfold to check if we don't have any overfitting
            result = self.kfold_model(text, X, y, 5, max_sentence_length, unique_syllables)
            print(result)

        if FLAGS.model_predict:
            
            train_texts = ['HEX_ELE-all.pickle']
            test_texts = ['SEN-aga.pickle']
            self.do_experiment(train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='seneca_precise', plot_title='Trimeter Texts')

            # train_texts = ['SEN-proofread.pickle']
            # test_texts = ['SEN-aga.pickle']
            # self.do_experiment(train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='seneca_precise', plot_title='Trimeter Texts')



        if FLAGS.exp_hexameter:

            train_texts = ['VERG-aene.pickle', 'IVV-satu.pickle', 'LVCR-rena.pickle', 'OV-meta.pickle', 'PERS-satu.pickle']
            test_texts = ['VERG-aene.pickle', 'IVV-satu.pickle', 'LVCR-rena.pickle', 'OV-meta.pickle', 'PERS-satu.pickle']

            # train_texts = ['VERG-aene.pickle', 'PROP-ele.pickle', 'OV-ele.pickle']
            # test_texts = ['VERG-aene.pickle', 'PROP-ele.pickle', 'OV-ele.pickle']

            # train_texts = ['PERS-satu.pickle','IVV-satu.pickle']
            # test_texts = ['PERS-satu.pickle','IVV-satu.pickle']

            sequence_labels_all_set = util.merge_sequence_label_lists(train_texts, util.cf.get('Pickle', 'path_sequence_labels')) # Merge them into one big file list
            all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
            # We need to extract the max sentence length over all these texts to get the padding correct later
            max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)       
            # And we need to create a list of all unique syllables for our word2idx one-hot encoding
            unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING)
            word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)  
            # twice test texts because they are identical and merge_sequence_label_lists has a bug
            self.do_experiment(test_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='hexameter', plot_title='Cross author evaluation')

        if FLAGS.exp_transfer_boeth:

            # To make the LSTM with integer hashing working, we need to make a list of all syllables from all the texts we are looking at    
            all_sequence_label_pickles = util.Create_files_list(util.cf.get('Pickle', 'path_sequence_labels'), 'pickle') # Find all pickle files
            sequence_labels_all_set = util.merge_sequence_label_lists(all_sequence_label_pickles, util.cf.get('Pickle', 'path_sequence_labels')) # Merge them into one big file list
            # all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
            # # We need to extract the max sentence length over all these texts to get the padding correct later
            # max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)       
            # text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), 'HEX_ELE-all.pickle')
            all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
            max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)
            # unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING) # needs to be sorted for word2idx consistency!

            # And we need to create a list of all unique syllables for our word2idx one-hot encoding
            unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING)
            word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)

            train_texts = ['VERG-aene.pickle', 'HEX-all.pickle', 'ELE-all.pickle', 'HEX_ELE-all.pickle']
            test_texts = ['BOETH-cons.pickle', 'TIB-ele.pickle', 'ENN-anna.pickle', 'HOR-arpo.pickle',
                          'LVCAN-phar.pickle', 'CATVLL-carm.pickle', 'STAT-theb.pickle']          
            
            self.do_experiment(train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='boethius', plot_title='Scanning Unseen Texts')

        if FLAGS.exp_transfer:
            # Here we test whether training on elegiac and hexameter gives better results
            train_texts = ['VERG-aene.pickle', 'HEX-all.pickle', 'ELE-all.pickle', 'HEX_ELE-all.pickle']
            test_texts = ['SEN-aga.pickle']
            self.do_experiment(train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='seneca', plot_title='Scanning Iambic Trimeter')

        if FLAGS.exp_elegiac:
            # Pick all the elegiac texts and let Virgil do his best :D
            train_texts = ['VERG-aene.pickle', 'TIB-ele.pickle', 'PROP-ele.pickle', 'OV-ele.pickle']
            test_texts = ['VERG-aene.pickle','TIB-ele.pickle', 'PROP-ele.pickle', 'OV-ele.pickle']
            self.do_experiment(train_texts, test_texts, exp_name='elegiac')

        if FLAGS.exp_train_test:
            train_texts = ['VERG-aene.pickle', 'IVV-satu.pickle', 'LVCR-rena.pickle', 'OV-meta.pickle']
            # train_texts = ['VERG-aene.pickle']

            for text in train_texts:
                current_text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), text)
                current_text = random.sample(current_text, 3600)

                all_text_syllables = self.retrieve_syllables_from_sequence_label_list(current_text)
                max_sentence_length = self.retrieve_max_sentence_length(current_text)
                unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING) # needs to be sorted for word2idx consistency!

                word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)

                X, y = self.create_X_y_sets(current_text, word2idx, label2idx, max_sentence_length)
             
                # we split in train and test set.
                X_test = X[:720]    # 20% of 3600
                y_test = y[:720]
                X_train = X[720:]
                y_train = y[720:]
                
                splits = 30

                X_train_list = np.array_split(X_train, splits)
                y_train_list = np.array_split(y_train, splits)

                X_train_list_2 = X_train_list[splits-1]
                y_train_list_2 = y_train_list[splits-1]

                # print(X_train_list[1])

                for i in range(splits):

                    # get the model we want
                    model = self.get_model( max_len = max_sentence_length,
                                            num_syllables = len(unique_syllables),
                                            num_labels = len(self.LABELS),
                                            X = X_train_list_2,
                                            y = y_train_list_2,
                                            epochs = self.num_epochs,
                                            create_model = True,
                                            save_model = False)
                    
                    loss, accuracy = self.evaluate_model(model, X_test, y_test)

                    result = '{0},{1},{2}\n'.format(text, accuracy, len(X_train_list_2))
                    # Open a file with access mode 'a'

                    file_name = './plots/size_' + text + '.txt'
                    file_object = open(file_name, 'a')
                    # Append 'hello' at the end of file
                    file_object.write(result)
                    # Close the file
                    file_object.close()

                    # Now increase the train list and repeat the experiment
                    X_train_list_2 = np.append(X_train_list_2, X_train_list[i], axis=0)
                    y_train_list_2 = np.append(y_train_list_2, y_train_list[i], axis=0)

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
                                    epochs = self.num_epochs,
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

    def run_idx_lstm_single_text(self, text, 
                                 do_evaluate=True, do_metric_report=True, do_confusion_matrix=True, print_stats=True):        
        # Retrieve meta data
        all_text_syllables = self.retrieve_syllables_from_sequence_label_list(text)
        max_sentence_length = self.retrieve_max_sentence_length(text)
        unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING) # needs to be sorted for word2idx consistency!

        # # Create dictionary
        word2idx = {w: i for i, w in enumerate(unique_syllables)}
        label2idx = {t: i for i, t in enumerate(self.LABELS)}
        # now we map the sentences and labels to a sequence of numbers
        X = [[word2idx[w[0]] for w in s] for s in text]  # key 0 are labels
        y = [[label2idx[w[1]] for w in s] for s in text] # key 1 are labels
        # and then (post)pad the sequences using the PADDING label.
        X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=word2idx[self.PADDING]) # value is our padding key
        y = pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=label2idx[self.PADDING])
        # for training the network we also need to change the labels to categorial.
        y = np.array([to_categorical(i, num_classes=len(self.LABELS)) for i in y])
        # we split in train and test set.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split_size)
        # get the model we want
        model = self.get_model( max_len = max_sentence_length,
                                num_syllables = len(unique_syllables),
                                num_labels = len(self.LABELS),
                                X = X_train,
                                y = y_train,
                                epochs = self.num_epochs,
                                create_model = True,
                                save_model = FLAGS.save_model)
        
        if do_evaluate:
            loss, accuracy = self.evaluate_model(model, X_test, y_test)
            result_string = 'RESULT: loss -> {0}, accuracy -> {1}. Length of test set: {2}\n'.format(loss, accuracy, len(y_test))
            print(result_string)
            # return loss, accuracy, len(y_train), len(y_test)

        if do_metric_report:
            metrics_report = self.create_metrics_report(model, X_test, y_test, output_dict=False)
            print(metrics_report)

        if do_confusion_matrix:
            confusion_matrix = self.create_confusion_matrix(model, X_test, y_test)
            df_confusion_matrix = pd.DataFrame(confusion_matrix, index = ['long', 'short', 'elision', 'space', 'padding'],
                                            columns = ['long', 'short', 'elision', 'space', 'padding'])
            # Drop the padding labels, as we don't need them (lstm scans them without confusion): delete both row and column
            df_confusion_matrix = df_confusion_matrix.drop('padding')
            df_confusion_matrix = df_confusion_matrix.drop(columns=['padding', 'space'])

            df_confusion_matrix = df_confusion_matrix.drop('space')
            # df_confusion_matrix = df_confusion_matrix.drop(columns=['padding'])

            util.create_heatmap(dataframe = df_confusion_matrix,
                                xlabel = 'TRUTH', 
                                ylabel =  'PREDICTED',
                                title = 'CONFUSION MATRIX',
                                filename = 'confusion_matrix_lstm_single_text',
                                # vmax = 500
                                ) 
        #################
        # TEMP PRINTING #
        #################
        if print_stats:
            # Keep track of wrong scansions
            sentence_incorrect_counter = 0
            syllable_incorrect_counter = 0
            
            # X = X_test
            # y = y_test

            # from collections import Counter
            # result = Counter(x for xs in y for x in set(xs))
            # print('RESULT', result)

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
                    syllable_incorrect_counter += (max_sentence_length - np.count_nonzero(y_pred_current==y_true_current))
                    # retrieve the syllables from the hashes so we can read the sentence again 11
                    syllable_list = []
                    for item in sentence:
                        syllable_list.append(list(word2idx.keys())[list(word2idx.values()).index(item)])                  
                    # now, for each line, print some information to investigate the problems
                    while 'PADDING' in syllable_list: syllable_list.remove('PADDING')    
                    # while 'PADDING' in syllable_list: syllable_list.remove('PADDING')    


                    print('syllables : ', syllable_list)
                    # print('prediction: ', y_pred_current)
                    # print('truth:      ', y_true_current)
                    # also, for latinists, print the scansion
                    y_pred_labels = ['—' if j==0 else '⏑' if j==1 else 'e' if j==2 else ' ' if j==3 else j for j in y_pred_current]
                    y_train_labels = ['—' if j==0 else '⏑' if j==1 else 'e' if j==2 else ' ' if j==3 else j for j in y_true_current]

                    while 4 in y_pred_labels: y_pred_labels.remove(4)    # padding.
                    while 4 in y_train_labels: y_train_labels.remove(4)    

                    print('prediction: ', y_pred_labels)
                    print('truth     : ', y_train_labels)
                    
                    print('\n##########################\n')
                    # count the sentence as incorrect
                    sentence_incorrect_counter += 1
                    


            # after all scrutiny, print the final statistics                    
            score_sentences = round(sentence_incorrect_counter/len(X)*100,2)
            # score_syllables = round(syllable_incorrect_counter/len(all_text_syllables)*100,2)

            print('SENTENCES SCANNED WRONGLY: ', sentence_incorrect_counter)
            print('PERCENTAGE WRONG: {0}%'.format(score_sentences))
            
            print('SYLLABLES SCANNED WRONGLY: ', syllable_incorrect_counter)
            # print('PERCENTAGE WRONG: {0}%'.format(score_syllables)) # This also counts spaces: needs fixing

    def do_experiment(self, train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='default', plot_title='default'):
        # import matplotlib.pyplot as plt        
        
        column_names = ["predictor", "predictee", "score"]
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
                                    epochs = self.num_epochs,
                                    create_model = FLAGS.create_model,
                                    save_model = FLAGS.save_model,
                                    model_name = train_text.split('.')[0])

            for test_text in test_texts:

                sequence_labels_test_text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), test_text)
                X_test, y_test = self.create_X_y_sets(sequence_labels_test_text, word2idx, label2idx, max_sentence_length)
                metrics_report = self.create_metrics_report(model, X_test, y_test)

                if FLAGS.metrics_report:
                    print_metrics_report = self.create_confusion_matrix(model, X_test, y_test)
                    
                    
                    print(print_metrics_report)
                    # print_metrics_report = self.create_metrics_report(model, X_test, y_test, output_dict=False)
                    # print(print_metrics_report)                    
                    
                    # exit(0)
                    self.create_seaborn_heatmap(print_metrics_report)

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
        X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=word2idx[self.PADDING]) # value is our padding key
        y = pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=label2idx[self.PADDING])
        # for training the network we also need to change the labels to categorial.
        y = np.array([to_categorical(i, num_classes=len(self.LABELS)) for i in y])

        return X, y

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

    def create_metrics_report(self, model, X, y, output_dict=True):
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

        if FLAGS.anceps: # We dont want to predict the label anceps, so we delete it from the possible predictions
            y_pred = y_pred[:, :, :-1]

        print(y_pred.shape)

        y_pred = np.argmax(y_pred, axis=-1)
        y = np.argmax(y, axis=-1)

        metrics_report = crf_metrics.flat_classification_report(
            y, y_pred, labels=[0,1,2], target_names=['long', 'short', 'elision'], digits=4, output_dict=output_dict
        )

        return metrics_report

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
        
        if create_model:
            # Initiate the model structure
            input = Input(shape=(max_len,))
            
            model = Embedding(input_dim=num_syllables, output_dim=50, input_length=max_len)(input)  # 50-dim embedding
            model = Dropout(0.1)(model)
            model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)  # variational biLSTM
            
            out = TimeDistributed(Dense(num_labels, activation="softmax"))(model)  # softmax output layer

            model = Model(input, out)

            # Compile the model
            model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
            history = model.fit(X, y, batch_size=32, epochs=epochs, verbose=self.FLAGS.verbose)
            
            if save_model: 
                model.save(path)

            return model        
        
        else:
            return keras.models.load_model(path)



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
        df_confusion_matrix = pd.DataFrame(confusion_matrix, index = ['long', 'short', 'elision', 'space', 'padding'],
                                        columns = ['long', 'short', 'elision', 'space', 'padding'])
        # Drop the padding labels, as we don't need them (lstm scans them without confusion): delete both row and column
        df_confusion_matrix = df_confusion_matrix.drop('padding')
        df_confusion_matrix = df_confusion_matrix.drop(columns=['padding', 'space'])

        df_confusion_matrix = df_confusion_matrix.drop('space')
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
    p.add_argument("--create_model", action="store_true", help="specify whether to create the model: if not specified, we load from disk")
    p.add_argument("--save_model", action="store_true", help="specify whether to save the model: if not specified, we do not save")
    p.add_argument("--single_text", action="store_true", help="specify whether to run the single text LSTM function")
    p.add_argument("--exp_hexameter", action="store_true", help="specify whether to run the hexameter LSTM experiment")
    p.add_argument("--exp_transfer", action="store_true", help="specify whether to run the hexameter transerability LSTM experiment")
    p.add_argument("--exp_elegiac", action="store_true", help="specify whether to run the hexameter genre LSTM experiment")
    p.add_argument("--exp_train_test", action="store_true", help="specify whether to run the train/test split LSTM experiment")
    p.add_argument("--exp_transfer_boeth", action="store_true", help="specify whether to run the Boeth LSTM experiment")
    p.add_argument("--model_predict", action="store_true", help="let a specific model predict a specific text")
    p.add_argument("--metrics_report", action="store_true", help="specifiy whether to print the metrics report")
    p.add_argument("--kfold", action="store_true", help="specifiy whether to run the kfold experiment")

    p.add_argument("--anceps", action="store_true", help="specify whether to train on an anceps text")
    p.add_argument("--verbose", action="store_true", help="specify whether to run the code in verbose mode")
    p.add_argument('--epochs', default=25, type=int, help='number of epochs')
    p.add_argument("--split", type=util.restricted_float, default=0.2, help="specify the split size of train/test sets")

    FLAGS = p.parse_args()    
    
    lstm = LSTM_model(FLAGS)