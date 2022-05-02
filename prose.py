'''
       ___       ___           ___     
      /\__\     /\  \         /\__\    
     /:/  /    /::\  \       /::|  |   
    /:/  /    /:/\ \  \     /:|:|  |   
   /:/  /    _\:\~\ \  \   /:/|:|__|__ 
  /:/__/    /\ \:\ \ \__\ /:/ |::::\__\ 
  \:\  \    \:\ \:\ \/__/ \/__/~~/:/  /
   \:\  \    \:\ \:\__\         /:/  / 
    \:\  \    \:\/:/  /        /:/  /  
     \:\__\    \::/  /        /:/  /   
      \/__/     \/__/         \/__/    
'''
# Latin Scansion Model
# Philippe Bors and Luuk Nolden
# Leiden University 2021

# Library Imports
import numpy as np
from progress.bar import Bar
import argparse
# from progress.bar import Bar

from cltk.prosody.lat.metrical_validator import MetricalValidator


import syllabifier as syl
import utilities as util

p = argparse.ArgumentParser(description='Argument parser for the LatinScansionModel')
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

class ProsePoetry():

    # text = "Scias ista, nescias: fient. Si vero solem ad rapidum4 stellasque sequentes ordine respicies, numquam te crastina fallet hora, nec insidiis noctis capiere serenae"
    # text = "Hoc scire quid proderit? ut sollicitus sim cum Saturnus et Mars ex contrario stabunt aut cum Mercurius vespertinum faciet occasum vidente Saturno, potius quam hoc discam, ubicumque sunt ista, propitia esse nec posse mutari? [15] Agit illa continuus ordo fatorum et inevitabilis cursus; per statas vices remeant et effectus rerum omnium aut movent aut notant. Sed sive quidquid evenit faciunt, quid inmutabilis rei notitia proficiet? sive significant, quid refert providere quod effugere non possis? Scias ista, nescias: fient. [16] Si vero solem ad rapidum stellasque sequentes ordine respicies, numquam te crastina fallet hora, nec insidiis noctis capiere serenae. Satis abundeque provisum est ut ab insidiis tutus essem. [17] 'Numquid me crastina non fallit hora? fallit enim quod nescienti evenit.' Ego quid futurum sit nescio: quid fieri possit scio. Ex hoc nihil deprecabor, totum expecto: si quid remittitur, boni consulo. Fallit me hora si parcit, sed ne sic quidem fallit. Nam quemadmodum scio omnia accidere posse, sic scio et non utique casura; itaque secunda expecto, malis paratus sum." 
    SPACE_CHAR = '-'
    
    min_syllables_hexameter = 12
    max_syllables_hexameter = 20
    
    do_lstm = True
    do_pedecerto = True
    do_anceps = True

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        # Read the prose text to search
        f = open("prose.txt", "r")
        text = f.read()
        f.close()
        # Clean it up
        text = self.clean_prose_text(text)
        text = text.split()

        # print(syllabified_text)
        print('Creating hexameter candidates from given prose text')
        possible_hexameter_list = self.create_hexameter_list_from_window(text, self.min_syllables_hexameter, self.max_syllables_hexameter)

        if self.do_lstm:
            # Let the LSTM think about what is acceptable as an hexameter
            # print('LSTM: creating candidate list')
            possible_hexameter_list = self.detect_hexameter_lstm(possible_hexameter_list)
            # print(possible_hexameter_list, len(possible_hexameter_list))            
        else:
            # Convert the syllabified list to Anceps & Pedecerto readable
            new_possible_hexameter_list = []
            for line in possible_hexameter_list:
                new_possible_hexameter_list.append(self.convert_split_hexameter_to_string(line))
            possible_hexameter_list = new_possible_hexameter_list

        if self.do_anceps:
            # Let Anceps think about what is acceptable as an hexameter
            print('Anceps: creating candidate list')
            possible_hexameter_list = self.detect_hexameter_anceps(possible_hexameter_list)
            # print(possible_hexameter_list, len(possible_hexameter_list))

        if self.do_pedecerto:
            print('Pedecerto: creating candidate list')
            possible_hexameter_list = self.detect_hexameter_pedecerto(possible_hexameter_list)
            # print(possible_hexameter_list, len(possible_hexameter_list))            

        print(' ')

        for line in possible_hexameter_list:
            print(line)
        print(len(possible_hexameter_list))

    def detect_hexameter_lstm(self, possible_hexameter_list):

        from lstm2 import LSTM_model
        from keras.preprocessing.sequence import pad_sequences
        from tensorflow import keras

        lstm = LSTM_model(self.FLAGS)
        hexameter_candidates = []

        poetry = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), 'VERG-aene.pickle')
        
        all_text_syllables = lstm.retrieve_syllables_from_sequence_label_list(poetry)
        max_sentence_length = lstm.retrieve_max_sentence_length(poetry)
        unique_syllables = np.append(sorted(list(set(all_text_syllables))), lstm.PADDING) # needs to be sorted for word2idx consistency!            
        word2idx, label2idx = lstm.create_idx_dictionaries(unique_syllables, lstm.LABELS)

        X, y = lstm.create_X_y_sets(poetry, word2idx, label2idx, max_sentence_length)

        model = lstm.get_model( max_len = max_sentence_length,
                                num_syllables = len(unique_syllables),
                                num_labels = len(lstm.LABELS),
                                X = X,
                                y = y,
                                epochs = 25,
                                create_model = False,
                                save_model = True,
                                model_name='Poetry')

        with Bar('LSTM: creating candidate list', max=len(possible_hexameter_list)) as bar:

            for possible_hexameter in possible_hexameter_list:
            # for possible_hexameter in Bar('LSTM: creating candidate list').iter(range(len(possible_hexameter_list))):
                try:
                    X = [word2idx[w] for w in possible_hexameter]# for s in possible_hexameter]  # key 0 are labels              
                    X = pad_sequences(maxlen=max_sentence_length, sequences=[X], padding="post", value=word2idx[lstm.PADDING]) # value is our padding key
                except:
                    bar.next()
                    continue
                # print(X)

                y_pred = model.predict(X)
                y_pred = np.argmax(y_pred, axis=-1)

                # y_labels = ['long' if i==0 else 'short' if i==1 else 'elision' if i==2 else 'space' if i==3 else 'padding' for i in y_pred[0]]
                # y_labels = ['—' if i==0 else '⏑' if i==1 else '∅' if i==2 else ' ' if i==3 else 'padding' for i in y_pred[0]]
                # y_labels = [x for x in y_labels if x != 'padding']
                # y_labels = ''.join([str(elem) for elem in y_labels])

                y_cltk = ['-' if i==0 else 'U' if i==1 else '' if i==2 else '' if i==3 else '' for i in y_pred[0]]
                y_cltk = ''.join([str(elem) for elem in y_cltk])
                
                if MetricalValidator().is_valid_hexameter(y_cltk):
                    # print('Line is hexameter')
                    # print(possible_hexameter, length_hexameter)  
                    # syllabified_line = ' '.join([str(elem) for elem in possible_hexameter])
                    original_line = self.convert_split_hexameter_to_string(possible_hexameter)
                    # original_line = ''.join([str(elem) for elem in possible_hexameter])
                    # original_line = original_line.replace('-',' ')
                    # print(original_line)
                    # print(syllabified_line)
                    # print(y_labels)
                    # print('')
                    hexameter_candidates.append(original_line)

                bar.next()


        return hexameter_candidates




    def detect_hexameter_anceps(self, hexameter_candidates):
        
        anceps_candidates = []
        
        from anceps.scan import Scan
        scan = Scan(hexameter_candidates)
        result = scan.result
        
        # import json
        # # Opening JSON file
        # f = open('./anceps/test.json')
        
        # # returns JSON object as
        # # a dictionary
        # result = json.load(f)
        
        # Iterating through the json
        # list
        for i in result['text']:
            if result['text'][i]['method'] != 'failed':
                anceps_candidates.append(result['text'][i]['verse'])
        
        # Closing file
        # f.close()

        # print(result)

        return anceps_candidates



    def detect_hexameter_pedecerto(self, hexameter_candidates):
        return hexameter_candidates

    def convert_split_hexameter_to_string(self, split_hexameter):
        original_line = ''.join([str(elem) for elem in split_hexameter])
        original_line = original_line.replace('-',' ')
        return original_line

    def create_hexameter_list_from_window(self, text, min_syllables_hexameter, max_syllables_hexameter):
        # Move with a window through the text
        possible_hexameter_list = []
        
        syllabified_text = self.syllabify_text(text)

        # Add a syllable to the beginning of the text: we will use this to detect if a syllable is the beginning of a word
        # Hexameters cant start with half a word.
        syllabified_text.insert(0, '-')

        with Bar('Creating candidate hexameters', max=len(syllabified_text)) as bar:
            while syllabified_text:
                length = self.calculate_hexameter_length(syllabified_text)
                # If we the amount of syllables left is smaller than min_syllables_hexameter, we cant make an hexameter anymore
                if length < self.min_syllables_hexameter:
                    break

                for i in range(min_syllables_hexameter, max_syllables_hexameter + 1):
                    # Quickly check if the given syllabified text is smaller than the length we are looking for.
                    # If so, just break the for loop.
                    if self.calculate_hexameter_length(syllabified_text) < i:
                        break

                    # Now take the first i syllables, but exclude the spaces in the counting
                    added_syllables = 0
                    possible_hexameter = []
                    
                    for element in syllabified_text:
                        # Stop if we have the number of syllables that we want
                        # If the syllable is a space, add it to detect word boundaries later
                        if added_syllables >= i:
                            if element == '-':
                                possible_hexameter.append(element)
                            break

                        if element != '-':
                            added_syllables += 1
                        
                        possible_hexameter.append(element)

                    # Check if the hexameter starts and ends with a word boundary
                    if self.check_word_boundaries(possible_hexameter):
                        # print(possible_hexameter, length_hexameter)
                        # Remove the starting and trailing whitespaces
                        possible_hexameter = possible_hexameter[1:-1]
                        possible_hexameter_list.append(possible_hexameter)
                # Remove the first syllable and continue the while
                syllabified_text.pop(0)
                bar.next()

                
        return possible_hexameter_list

    def syllabify_text(self, text):
        # Syllabify the text and add it to a single list
        syllabified_text = []
        for idx, word in enumerate(text):
            syllabified = syl.syllabify(word)
            syllabified.append('-')
            syllabified_text += syllabified

        return syllabified_text

    def clean_prose_text(self, text):
        # Strip the text of non alphanumerical characters, lower it and merge whitespaces
        text = text.translate({ord(c): None for c in ",.:?![]';()"})
        text = text.translate({ord(c): None for c in '0123456789'})
        text = text.lower()
        text = ' '.join(text.split())
        return text

    def check_word_boundaries(self, possible_hexameter_line):
        # An hexameter only works on word boundaries. this function checks that        
        return (possible_hexameter_line[0] == self.SPACE_CHAR and possible_hexameter_line[-1] == self.SPACE_CHAR)
        #     return True
        # else:
        #     return False

    def calculate_hexameter_length(self, given_hexameter):
        return len([value for value in given_hexameter if value != '-'])

if __name__ == "__main__":
    prose = ProsePoetry(FLAGS)


    # X, y = lstm.create_X_y_sets(poetry, word2idx, label2idx, max_sentence_length)

    

    # exit(0)      
        
