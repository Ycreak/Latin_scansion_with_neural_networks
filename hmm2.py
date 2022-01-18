# Based on: https://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint 

from sklearn.metrics import precision_recall_fscore_support
from sklearn_crfsuite import metrics

import utilities as util
import configparser

from progress.bar import Bar

class Hidden_markov_model:

    cf = configparser.ConfigParser()
    cf.read("config.ini")

    perform_sentence_transition_list = False
    perform_matrix_alpha_creation = False
    perform_matrix_beta_creation = False
    perform_viterbi_walks = False

    pedecerto_df = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'pedecerto_df'))
    label_list = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'label_list'))
    
    def __init__(self):

        # Load the pedecerto df and convert its integer labels to strings
        self.pedecerto_df = self.pedecerto_df_labels_to_str(self.pedecerto_df)        

        # Create hidden state space (our labels)
        hidden_states = ['long', 'short', 'elision']

        # create hidden transition matrix alpha
        # this is the transition probability matrix of changing states given a state
        # matrix is size (M x M) where M is number of states

        # create state space and initial state probabilities
        if self.perform_sentence_transition_list:
            self.label_list = self.create_sentence_transition_list(self.pedecerto_df)
            util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'label_list'), self.label_list)

        if self.perform_matrix_alpha_creation:
            a_df = self.create_hidden_transition_matrix_alpha(hidden_states, self.label_list)
            util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_a'), a_df)

        a_df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_a'))
        if self.cf.get('Util', 'verbose'): print(a_df)

        # create matrix of observation (emission) probabilities beta.
        # this holds the observation probabilities given state.
        # matrix is size (M x O) where M is number of states 
        # and O is number of different possible observations.
        unique_syllables = sorted(set(self.pedecerto_df['syllable'].tolist()))
        observable_states = unique_syllables # Our observations are all of our unique syllables

        if self.perform_matrix_beta_creation:
            b_df = self.create_hidden_transition_matrix_beta(observable_states, hidden_states, unique_syllables, self.pedecerto_df)
            util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_b'), b_df)

        b_df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_b'))
        if self.cf.get('Util', 'verbose'): print(b_df)

        custom_sentence = "li to ra mul tum il le et ter ris jac ta tus et al to"
        custom_sentence = "ar ma vi rum que ca no troi ae qui pri mus ab or is"

        # Get parameters ready for the viterbi walks
        pi = self.get_label_probabilities(self.pedecerto_df)
        a = a_df.values
        b = b_df.values

        if self.perform_viterbi_walks:
            y_true, y_pred = self.create_y(self.pedecerto_df, observable_states, pi, a, b)
            util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_y_pred'), y_pred)
            util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_y_true'), y_true)

        y_true = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_y_true'))
        y_pred = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_y_pred'))

        self.create_prediction_df(y_true, y_pred)

        print('##########################################################')
        print(self.get_metrics_report(y_true, y_pred))
        print('##########################################################')



    def create_prediction_df(self, df, y, yhat):
        df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'flattened_vectors'))
        # Creates a dataframe with predictions. Used by OSCC (for now)
        column_names = ["predicted", "expected"]
        new_df = pd.DataFrame(columns = column_names)

        for i in Bar('Processing').iter(range(len(y))):
            new_line = {'expected': y[i], 'predicted': yhat[i]}
            new_df = new_df.append(new_line, ignore_index=True)

        book_line_df = df[['book','line', 'syllable']]

        prediction_df = pd.concat([book_line_df, new_df], axis=1, join='inner')

        print(prediction_df)

        util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_prediction_df'), prediction_df)

    def create_y(self, pedecerto_df, observable_states, pi, a, b):
        y_pred = []
        y_true = []
        # Get number of books to process
        num_books = pedecerto_df['book'].max()
        for i in Bar('Processing').iter(range(num_books)):
            # Get only lines from this book
            current_book = i + 1
            book_df = pedecerto_df.loc[pedecerto_df['book'] == current_book]

            num_lines = book_df['line'].max()

            for j in range(num_lines): # now process all lines using the HMM
                current_line = j + 1
                filtered_df = book_df[book_df["line"] == current_line]
                
                true_label_list = filtered_df['length'].tolist()
                syllable_list = filtered_df['syllable'].tolist()
                
                if syllable_list: #FIXME: this is bad
                    result = self.predict_single_sentence(syllable_list, observable_states, pi, a, b)
                    y_pred.append(result['Best_Path'].tolist())
                    y_true.append(true_label_list)
                
        return y_true, y_pred

    def get_metrics_report(self, y_true, y_pred):
        sorted_labels = sorted(
            ['long', 'short', 'elision'],
            key=lambda name: (name[1:], name[0])
        )
        metrics_report = metrics.flat_classification_report(
            y_true, y_pred, labels=sorted_labels, digits=3
        )        
        return metrics_report

    def predict_single_sentence(self, syllable_list, observable_states, pi, a, b):
        debug_mode = False
        
        sentence_array = np.array([])

        for syllable in syllable_list: # Convert syllable list to observable_state list (using its indeces)
            sentence_array = np.append(sentence_array, observable_states.index(syllable))

        obs = sentence_array.astype(int)

        obs_map = {}
        for state in observable_states:
            obs_map[state] = observable_states.index(state)

        inv_obs_map = dict((v,k) for k, v in obs_map.items())
        obs_seq = [inv_obs_map[v] for v in list(obs)]

        # Sequence of overservations (and their code)
        if debug_mode:
            print( pd.DataFrame(np.column_stack([obs, obs_seq]), 
                        columns=['Obs_code', 'Obs_seq']) )

        # Do the viterbi walk
        path, delta, phi = self.viterbi(pi, a, b, obs)
        
        if debug_mode:
            print('\nsingle best state path: \n', path)
            print('delta:\n', delta)
            print('phi:\n', phi)

        state_map = {0:'long', 1:'short',2:'elision'}
        state_path = [state_map[v] for v in path]

        result = ((pd.DataFrame()
        .assign(Observation=obs_seq)
        .assign(Best_Path=state_path)))

        if debug_mode: print(result)

        return result

    def pedecerto_df_labels_to_str(self, df):
        df['length'] = np.where(df['length'] == 0, 'short', df['length'])
        df['length'] = np.where(df['length'] == 1, 'long', df['length'])
        df['length'] = np.where(df['length'] == 2, 'elision', df['length'])
        return df
    
    # define Viterbi algorithm for shortest path
    # code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
    # https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py
    def viterbi(self, pi, a, b, obs):
        
        debug_mode = False

        nStates = np.shape(b)[0]
        T = np.shape(obs)[0]
        
        # init blank path
        path = path = np.zeros(T,dtype=int)
        # delta --> highest probability of any path that reaches state i
        delta = np.zeros((nStates, T))
        # phi --> argmax by time step for each state
        phi = np.zeros((nStates, T))
        
        # init delta and phi 
        delta[:, 0] = pi * b[:, obs[0]]
        phi[:, 0] = 0

        if debug_mode: print('\nStart Walk Forward\n')    
        # the forward algorithm extension
        for t in range(1, T):
            for s in range(nStates):
                delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
                phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
                if debug_mode: print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
        
        # find optimal path
        if debug_mode: print('-'*50)
        if debug_mode: print('Start Backtrace\n')
        path[T-1] = np.argmax(delta[:, T-1])
        for t in range(T-2, -1, -1):
            path[t] = phi[path[t+1], [t+1]]
            if debug_mode: print('path[{}] = {}'.format(t, path[t]))
            
        return path, delta, phi

    def create_sentence_transition_list(self, df) -> list:
        
        all_sentences_list = []
        # Get number of books to process
        num_books = df['book'].max()
        for i in Bar('Processing').iter(range(num_books)):
            # Get only lines from this book
            current_book = i + 1
            book_df = df.loc[df['book'] == current_book]

            num_lines = book_df['line'].max()

            for j in range(num_lines):
                current_line = j + 1
                filtered_df = book_df[book_df["line"] == current_line]
                length_list = filtered_df['length'].tolist()
                all_sentences_list.append(length_list)

        return all_sentences_list

    def create_hidden_transition_matrix_alpha(self, hidden_states, label_list):
    # Now we are going to fill the hidden transition matrix
        a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)

        ll = 0
        ls = 0
        le = 0
        sl = 0
        ss = 0
        se = 0
        el = 0
        es = 0
        ee = 0

        total_count = 0

        for sentence in label_list:
            
            syllable_count = len(sentence)
            
            for idx, syllable in enumerate(sentence):

                if idx+1 < syllable_count:

                    item1 = sentence[idx]
                    item2 = sentence[idx+1]
            
                    if item1 == 'long' and item2 == 'long':ll +=1
                    elif item1 == 'long' and item2 == 'short': ls +=1
                    elif item1 == 'long' and item2 == 'elision': le +=1
                    elif item1 == 'short' and item2 == 'long': sl +=1
                    elif item1 == 'short' and item2 == 'short': ss +=1
                    elif item1 == 'short' and item2 == 'elision': se +=1
                    elif item1 == 'elision' and item2 == 'long': el +=1
                    elif item1 == 'elision' and item2 == 'short': es +=1
                    elif item1 == 'elision' and item2 == 'elision': ee +=1
                    else:
                        raise Exception('unknown transition found')

                else:
                    break

            total_count += syllable_count -1

            # print(syllable_count)
            # exit(0)

        prob_ll = ll/total_count
        prob_ls = ls/total_count
        prob_le = le/total_count
        prob_sl = sl/total_count
        prob_ss = ss/total_count
        prob_se = se/total_count
        prob_el = el/total_count 
        prob_es = es/total_count
        prob_ee = ee/total_count


        a_df.loc[hidden_states[0]] = [prob_ll, prob_ls, prob_le]
        a_df.loc[hidden_states[1]] = [prob_sl, prob_ss, prob_se]
        a_df.loc[hidden_states[2]] = [prob_el, prob_es, prob_ee]

        return a_df

    def create_hidden_transition_matrix_beta(self, observable_states, hidden_states, unique_syllables, pedecerto_df):

        b_df = pd.DataFrame(columns=observable_states, index=hidden_states)

        total_syllable_count = len(pedecerto_df)

        for syllable in unique_syllables:
            filtered_df = pedecerto_df[pedecerto_df["syllable"] == syllable]
            
            filter = filtered_df['length'].value_counts()

            try:    
                b_df.at['long',syllable]    =filter['long']/total_syllable_count
            except:
                pass
            try:
                b_df.at['short',syllable]   =filter['short']/total_syllable_count
            except:
                pass

            try:
                b_df.at['elision',syllable] =filter['elision']/total_syllable_count
            except:
                pass

        b_df = b_df.fillna(0)

        return b_df

    def get_label_probabilities(self, pedecerto_df):

        filter = pedecerto_df['length'].value_counts()

        long = filter['long']/len(pedecerto_df)
        short = filter['short']/len(pedecerto_df)
        elision = filter['elision']/len(pedecerto_df)

        return [long, short, elision]

hmm = Hidden_markov_model()
