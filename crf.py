# Library imports
import pandas as pd
import numpy as np
from progress.bar import Bar
import seaborn as sn
import matplotlib.pyplot as plt

# CRF specific imports
import scipy.stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

# Class imports
import utilities as util
from pedecerto.textparser import Pedecerto_parser

class CRF_sequence_labeling:
    ''' This class handels Conditional Random Fields sequence labeling.
    '''
    
    # feature-based sequence labelling: conditional random fields
    labels = ['short', 'long', 'elision']

    def __init__(self):
        pass

    def perform_experiments(self):    
        # self.elegiac_heatmap()
        self.hexameter_heatmap()
        # self.crf_improvement_heatmap()

    def convert_pedecerto_dataframes_to_sequence_labeling_list(self, source, destination):
        """Converts all pedecerto dataframes in the given location to sequence labeling lists.
        Saves these to disk in the specified location.

        Args:
            source (string): source location of the pedecerto dataframes
            destination (string): destination location of the pedecerto dataframes
        """        
        texts = util.Create_files_list(source, '.pickle')
        for text in texts:
            df = util.Pickle_read(source, text)
            # Convert the integer labels to string labels (like sequence labeling likes)
            df = util.convert_syllable_labels(df)
            # convert the current file to a sequence labeling list
            sequence_label_list = self.convert_pedecerto_to_sequence_labeling(df)
            # extract the name of the file to be used for pickle saving
            text_name = text.split('.')[0]
            text_name = text.split('_')[-1]
            # And write it to the location specified
            util.Pickle_write(destination, text_name, sequence_label_list)

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
            self.labels,
            key=lambda name: (name[1:], name[0])
        )
        return crf_metrics.flat_classification_report(
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
            # '0:last_3_char': word[-2:], # Take last 2 characters
            # '0:last_2_char': word[-1:], # Take last 1 characters
        }
        # Check if we are at the beginning of the sentence
        if i == 0:
            features['BOS'] = True # This should always be long
        # Gather features from the previous word
        if i > 0:
            previous_word = sent[i-1][0]
            features.update({
                '-1:word': previous_word,
                # '-1:last_1_char': previous_word[-1:],
                # '-1:last_2_char': previous_word[-2:],
            })
        # Gather features from the next word
        if i < len(sent)-1:
            next_word = sent[i+1][0]
            features.update({
                '+1:word': next_word,
                # '+1:first_1_char': next_word[:1],
                # '+1:first_2_char': next_word[:2],
            })
        else:
            features['EOS'] = True # This will be an anceps

        return features

    def convert_pedecerto_to_sequence_labeling(self, df) -> list:
        """Converts the given pedecerto dataframe to a list with sequence labels. More specifically,
        one list with multiple lists is returned. Each sublist represents a sentence with syllable and label.
        Such sublist looks as follows: [(syllable, label),(syllable, label), (syllable, label)]

        Args:
            df (dataframe): of a text in the pedecerto format

        Returns:
            list: with sequence labels (to serve as input for sequence labeling tasks)
        """              
        # Create a list to store all texts in
        all_sentences_list = []
        # get the integers for all titles to loop through
        all_titles = df['title'].unique()
        for title in Bar('Converting Pedecerto to CRF').iter(all_titles):
            # Get only lines from this book
            title_df = df.loc[df['title'] == title]
            # Per book, process the lines
            all_lines = title_df['line'].unique()
            for line in all_lines:
                line_df = title_df[title_df["line"] == line]

                length_list = line_df['length'].to_numpy()
                syllable_list = line_df['syllable'].to_numpy()
                # join them into 2d array and transpose it to get the correct crf format:
                combined_list = np.array((syllable_list,length_list)).T
                # Append all to the list which we will return later
                all_sentences_list.append(combined_list)

        return all_sentences_list

    def kfold_model(self, sequence_labels, X, y, splits):
        """Performs a kfold cross validation of a CRF model fitted and trained on the given data

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
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)
        kf = KFold(n_splits=splits, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(sequence_labels):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            crf = self.fit_model(X_train, y_train)

            metrics_report = self.predict_model(crf, X_test, y_test)
            # TODO: just print a score for each run to terminal
            report_list.append(metrics_report)

        result = self.merge_kfold_reports(report_list)

        return result

    def merge_crf_dataframes(self, texts, path):
        """Merges the given lists (contained in sequence labeling pickles) on the given path.
        Outputs one list with all sentences of the given texts in sequence labeling format.
        Useful when merging all metamorphoses for example.

        Args:
            texts (list): of sequence labeling pickled files
            path (string): where these pickled files are stored

        Returns:
            list: of merged texts
        """        
        # Create a starting list from the last entry using pop
        merged_list = util.Pickle_read(path, texts.pop())
        # merge all other texts into this initial list
        for text_list_id in texts:
            # from the list with texts
            text_list = util.Pickle_read(path, text_list_id)
            # take every sentence and add it to the merged_list
            for sentence_numpy in text_list:
                merged_list.append(sentence_numpy)
        return merged_list 

    ###############
    # EXPERIMENTS #
    ###############
    def elegiac_heatmap(self):

        column_names = ["predictor", "predictee", "score"]
        df_long = pd.DataFrame(columns = column_names)
        df_short = pd.DataFrame(columns = column_names)
        df_elision = pd.DataFrame(columns = column_names)

        ovid_elegiac_df = self.merge_crf_dataframes(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), 'OV'), util.cf.get('Pickle', 'df_crf_path_elegiac'))
        tib_elegiac_df = self.merge_crf_dataframes(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), 'TIB'), util.cf.get('Pickle', 'df_crf_path_elegiac'))
        prop_elegiac_df = self.merge_crf_dataframes(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), 'PROP'), util.cf.get('Pickle', 'df_crf_path_elegiac'))

        verg_df = util.Pickle_read(util.cf.get('Pickle', 'df_crf_path_hexameter'), 'VERG-aene.pickle')
        hexameter_df = self.merge_crf_dataframes(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_hexameter'), '.pickle'), util.cf.get('Pickle', 'df_crf_path_hexameter'))
        elegiac_df = self.merge_crf_dataframes(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), '.pickle'), util.cf.get('Pickle', 'df_crf_path_elegiac'))
        
        # First, use Aneid to predict Ovid and Tibullus
        predictor_texts = [('verg',verg_df), ('hex',hexameter_df), ('eleg',elegiac_df)]
        predictee_texts = [('ovid',ovid_elegiac_df), ('tib',tib_elegiac_df), ('prop',prop_elegiac_df)]

        for predictor_text in predictor_texts:
            predictor_X, predictor_y = self.convert_text_to_feature_sets(predictor_text[1])
            crf_model = self.fit_model(predictor_X, predictor_y)

            for predictee_text in predictee_texts:
                predictee_X, predictee_y = self.convert_text_to_feature_sets(predictee_text[1])
                # Using the predictor model, predict for the predictee test set
                result = self.predict_model(crf_model, predictee_X, predictee_y)

                predictor = predictor_text[0] # get names
                predictee = predictee_text[0]

                print(predictor, predictee)
                # exit(0)

                score_long = float(round(result['long']['f1-score'] * 100,1)) # save the score
                score_short = float(round(result['short']['f1-score'] * 100,1)) # save the score
                score_elision = float(round(result['elision']['f1-score'] * 100,1)) # save the score
                
                # Add score to the dataframe for our heatmap
                new_line_long = {'predictor': predictor, 'predictee': predictee, 'score': score_long}
                new_line_short = {'predictor': predictor, 'predictee': predictee, 'score': score_short}
                new_line_elision = {'predictor': predictor, 'predictee': predictee, 'score': score_elision}

                df_long = df_long.append(new_line_long, ignore_index=True)    
                df_short = df_short.append(new_line_short, ignore_index=True)    
                df_elision = df_elision.append(new_line_elision, ignore_index=True)

        # pivot the dataframe to be usable with the seaborn heatmap
        heatmap_data = pd.pivot_table(df_long, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- long syllables',
                        filename = 'confusionmatrix_elegiac_long.png')

        heatmap_data = pd.pivot_table(df_short, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- short syllables',
                        filename = 'confusionmatrix_elegiac_short.png')

        heatmap_data = pd.pivot_table(df_elision, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- elided syllables',
                        filename = 'confusionmatrix_elegiac_elision.png')
                        
        # exit(0)


    def hexameter_heatmap(self):

        ############################
        # CREATE HEXAMETER HEATMAP #
        ############################
        column_names = ["predictor", "predictee", "score"]
        df_long = pd.DataFrame(columns = column_names)
        df_short = pd.DataFrame(columns = column_names)
        df_elision = pd.DataFrame(columns = column_names)

        predictor_texts = util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_hexameter'), '.pickle')
        predictee_texts = util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_hexameter'), '.pickle')

        # Let every text be the predictor once
        for predictor_text in predictor_texts:
            predictor_df = util.Pickle_read(util.cf.get('Pickle', 'df_crf_path_hexameter'), predictor_text)
            predictor_X, predictor_y = self.convert_text_to_feature_sets(predictor_df)
            crf_model = self.fit_model(predictor_X, predictor_y)

            # For every predictor, get results for every predictee. This includes predicting itself
            for predictee_text in predictee_texts:
                predictee_df = util.Pickle_read(util.cf.get('Pickle', 'df_crf_path_hexameter'), predictee_text)
                predictee_X, predictee_y = self.convert_text_to_feature_sets(predictee_df)
                # Using the predictor model, predict for the predictee test set
                result = self.predict_model(crf_model, predictee_X, predictee_y)

                predictor = predictor_text.split('-')[0].capitalize() # get names
                predictee = predictee_text.split('-')[0].capitalize()
                score_long = float(round(result['long']['f1-score'] * 100,1)) # save the score
                score_short = float(round(result['short']['f1-score'] * 100,1)) # save the score
                score_elision = float(round(result['elision']['f1-score'] * 100,1)) # save the score
                
                # Add score to the dataframe for our heatmap
                new_line_long = {'predictor': predictor, 'predictee': predictee, 'score': score_long}
                new_line_short = {'predictor': predictor, 'predictee': predictee, 'score': score_short}
                new_line_elision = {'predictor': predictor, 'predictee': predictee, 'score': score_elision}

                df_long = df_long.append(new_line_long, ignore_index=True)    
                df_short = df_short.append(new_line_short, ignore_index=True)    
                df_elision = df_elision.append(new_line_elision, ignore_index=True)    

        # df_long.to_csv('./csv/long_1_char_only.csv')
        # df_short.to_csv('./csv/short_1_char_only.csv')
        # df_elision.to_csv('./csv/elision_1_char_only.csv')
        
        # pivot the dataframe to be usable with the seaborn heatmap
        heatmap_data = pd.pivot_table(df_long, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- long syllables',
                        filename = 'confusionmatrix_hexameter_long.png')

        heatmap_data = pd.pivot_table(df_short, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- short syllables',
                        filename = 'confusionmatrix_hexameter_short.png')


        heatmap_data = pd.pivot_table(df_elision, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- elided syllables',
                        filename = 'confusionmatrix_hexameter_elision.png')

    def create_heatmap(self, dataframe, xlabel, ylabel, title, filename):
        # Simple function to create a heatmap
        sn.set(font_scale=1.4)
        sn.heatmap(dataframe, annot=True, fmt='g', annot_kws={"size": 16}, cmap='Blues')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(filename, bbox_inches='tight')        
        plt.clf()    

if __name__ == "__main__":
    crf = CRF_sequence_labeling()
    crf.hexameter_heatmap()
