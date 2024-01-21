# Third party library imports
import numpy as np
import argparse

import sklearn
import sklearn.model_selection
# import scikit-learn as sklearn

import sklearn_crfsuite

# Class imports
from lsnn import utilities as util

class Latin_CRF:
    ''' This class handels Conditional Random Fields sequence labeling.
    '''
    
    # feature-based sequence labelling: conditional random fields
    LABELS = ['short', 'long', 'elision']

    def remove_space_from_syllable_sequence(self, given_list):
        """Removes the space labels from the syllable sequence lists, as the CRFs performance
        is deteriorated by the spaces.

        Args:
            given_list (list): with sequence labels

        Returns:
            list: now without space labels
        """        
        new_list = []

        for my_list in given_list:
            out_tup = [i for i in my_list if i[1] != 'space']
            new_list.append(out_tup)

        return new_list

    def custom_prediction(self, predictor, predictee):
        """Does a custom prediction by training on the predictor sequence label list and testing
        on the predictee sequence list

        Args:
            predictor_pickle (list): predictor list to function as trainer
            predictee_pickle (list): predictee list that is to be predicted by the trainer
        """        
        predictor_text = self.remove_space_from_syllable_sequence(predictor)
        predictee_text = self.remove_space_from_syllable_sequence(predictee)

        predictor_X, predictor_y = self.convert_text_to_feature_sets(predictor_text)
        crf_model = self.fit_model(predictor_X, predictor_y)

        predictee_X, predictee_y = self.convert_text_to_feature_sets(predictee_text)
        result = self.predict_model(crf_model, predictee_X, predictee_y)

        return result

    def predict_model(self, model, X, y):
        """ Predicts labels y given the model and examples X. Returns the metric report.

        Args:
            model (crf_model): model trained
            X (numpy array): with training or testing examples
            y (numpy array): with labels

        Returns:
            metrics_report: with model score
        """        
        y_pred = model.predict(X)
        #TODO: this gives errors
        sorted_labels = sorted(
            self.LABELS,
            key=lambda name: (name[1:], name[0])
        )
        return sklearn_crfsuite.metrics.flat_classification_report(
            y, y_pred, labels=sorted_labels, output_dict=True)

    def fit_model(self, X, y) -> object:
        """Creates a fitted model given X and y.

        Args:
            X (numpy array): of examples
            y (numpy array): of labels

        Returns:
            crf model: to be used for predicting
        """        
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X, y)
        return crf

    def convert_text_to_feature_sets(self, syllable_label_list):
        """Intermediate function to turn a dataframe with syllables and labels into proper
        X and y training/test sets. Given is a list with a syllable and its label, returned are
        two sets, one with syllables and their features (called X) and one with labels (called y).

        Args:
            syllable_label_list (list): of syllables and their labels

        Returns:
            lists: X and y sets to train/test the model on
        """
        X = [self.sentence_to_features(s) for s in syllable_label_list]
        y = [self.sentence_to_labels(s) for s in syllable_label_list]

        return X, y

    def sentence_to_labels(self, sent):
        """Given a sentence, this function will return a list of all labels in the given sentence.
        Used to create an y training/test set

        Args:
            sent (string): of a sentence with words and labels

        Returns:
            list: of sentence, now with each word accompanied by a feature dictionary
        """     
        return [label for token, label in sent]

    def sentence_to_features(self, sent):
        """Given a sentence, this function will add features to each of its words

        Args:
            sent (string): of a sentence with words and labels

        Returns:
            list: of sentence, now with each word accompanied by a feature dictionary
        """        
        return [self.add_word_features(sent, i) for i in range(len(sent))]

    def add_word_features(self, sent, i):
        """Given a sentence and the word index, add a feature dictionary to
        each word that can help with predicting its label. 
        NB: A word here is used for each syllable!

        Args:
            sent (string): with the entire sentence
            i (int): index to find the current word

        Returns:
            features: a set of features for the inspected word/syllable
        """        
        # Save the current word/syllable
        word = sent[i][0]
        # First, create features about the current word
        features = {
            'bias': 1.0,
            '0:syllable' : word, # just take the entire syllable
            '0:last_3_char': word[-2:], # Take last 2 characters
            '0:last_2_char': word[-1:], # Take last 1 characters
        }
        # Check if we are at the beginning of the sentence
        if i == 0:
            features['BOS'] = True # This should always be long
        # Gather features from the previous word
        if i > 0:
            previous_word = sent[i-1][0]
            features.update({
                '-1:word': previous_word,
                '-1:last_1_char': previous_word[-1:],
                '-1:last_2_char': previous_word[-2:],
            })
        # Gather features from the next word
        if i < len(sent)-1:
            next_word = sent[i+1][0]
            features.update({
                '+1:word': next_word,
                '+1:first_1_char': next_word[:1],
                '+1:first_2_char': next_word[:2],
            })
        else:
            features['EOS'] = True # This will be an anceps

        return features

    def kfold_model(self, sequence_labels : list, splits : int = 5):
        """Performs a kfold cross validation of a CRF model fitted and trained on the given data

        Args:
            sequence_labels (list): of sequence labels and their features

        Returns:
            dict: with results of the cross validation
        """        
        sequence_labels = self.remove_space_from_syllable_sequence(sequence_labels)
        X, y = self.convert_text_to_feature_sets(sequence_labels)

        report_list = []

        # Convert the list of numpy arrays to a numpy array with numpy arrays
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)
        kf = sklearn.model_selection.KFold(n_splits=splits, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(sequence_labels):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            crf = self.fit_model(X_train, y_train)

            metrics_report = self.predict_model(crf, X_test, y_test)
            # TODO: just print a score for each run to terminal
            report_list.append(metrics_report)

        result = util.merge_kfold_reports(report_list)

        return result

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--custom_prediction", action="store_true", help="specify whether to create the model: if not specified, we load from disk")
#     p.add_argument("--kfold", action="store_true", help="specify whether to run kfold experiment")
#     FLAGS = p.parse_args()    

#     latin_crf = Latin_CRF()

#     if FLAGS.custom_prediction:
#         latin_crf.custom_prediction(
#             predictor = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, FLAGS.train),
#             predictee = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, FLAGS.test))

#     if FLAGS.kfold:
#         result = latin_crf.kfold_model(
#             sequence_labels = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, FLAGS.train),
#             splits = 5)
#         print(result)

