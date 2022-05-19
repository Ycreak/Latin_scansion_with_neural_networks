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

p.add_argument("--lstm", action="store_true", help="specifiy whether to run the kfold experiment")
p.add_argument("--pedecerto", action="store_true", help="specifiy whether to run the kfold experiment")
p.add_argument("--anceps", action="store_true", help="specify whether to train on an anceps text")
p.add_argument("--fuzzywuzzy", action="store_true", help="specify whether to train on an anceps text")
p.add_argument("--candidates", action="store_true", help="specify whether to train on an anceps text")

p.add_argument("--verbose", action="store_true", help="specify whether to run the code in verbose mode")
p.add_argument('--epochs', default=25, type=int, help='number of epochs')
p.add_argument("--split", type=util.restricted_float, default=0.2, help="specify the split size of train/test sets")

FLAGS = p.parse_args()

class ProsePoetry():

    SPACE_CHAR = '-'
    
    min_syllables_hexameter = 12
    max_syllables_hexameter = 20

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        do_candidates = FLAGS.candidates
        do_lstm = FLAGS.lstm
        do_anceps = FLAGS.anceps
        do_pedecerto = FLAGS.pedecerto
        do_fuzzywuzzy = FLAGS.fuzzywuzzy

        prose_path = './prose/'
        prose_text = 'macrobius'
        prose_clean = prose_path + prose_text + '_clean.txt'
        prose_dirty = prose_path + prose_text + '.txt'
        prose_punctuation = prose_path + prose_text + '_punctuation.txt'

        # Read the prose text to search
        all_text = self.read_file(prose_clean)
        # print(all_text)
        # exit(0)

        if do_fuzzywuzzy:
            # In this function, we take all our candidates and map them on the original text for qualitative analysis
            # Notice that for this function, we'd like to take the original text to map onto!
            import re
            import string
            import collections

            debugging = False

            # Create two texts, the first one being a clean version            
            # cleaned_text = all_text.translate({ord(c): None for c in "*[]'()~<>.,?!:;"})
            # cleaned_text = cleaned_text.translate({ord(c): None for c in '0123456789'})
            # cleaned_text = cleaned_text.replace('\n'," ").lower()#            all_text = all_text.strip('\n')
            cleaned_text = self.read_file(prose_clean).split()
            # And the second one being one with punctuation
            text_with_punctuation = self.read_file(prose_punctuation)
            # text_with_punctuation = text_with_punctuation.translate({ord(c): None for c in "*[]()~<>"})
            # text_with_punctuation = text_with_punctuation.translate({ord(c): None for c in '0123456789'})
            text_with_punctuation = text_with_punctuation.split()

            # print(len(cleaned_text))
            # print(len(text_with_punctuation))

            # for idx, item in enumerate(cleaned_text):
            #     print(idx, item, text_with_punctuation[idx])

            # print(text_with_punctuation)
            # exit(0)

            candidates_path = './prose/' + prose_text + '_pedecerto_candidates.txt'
            candidates_file = open(candidates_path, 'r')

            # with open(candidates_path) as file:
            #     lines = file.readlines()
            #     lines = [line.rstrip() for line in lines]

            # print(len(lines))
            # print(len(set(lines)))
            # exit(0)

            text_with_punctuation2 = ' '.join(text_with_punctuation)
            cleaned_text2 = ' '.join(cleaned_text)

            # lowest_number = 1000000
            # my_string_saved = ''
            # my_string_saved2 = ''

            for candidate_line in candidates_file:
                candidate_line = candidate_line.strip().split()
                
                try:
                    begin, end = self.find_sub_list(candidate_line, cleaned_text)
                    
                    if debugging: print('YOUR CANDIDATE: ', candidate_line)
                    if debugging: print('IN THE TEXT: ', text_with_punctuation[begin:end+1])

                    # x = [''.join(c for c in s if c not in string.punctuation) for s in text_with_punctuation[begin:end+1]]   

                    # # print(x)
                    # if not collections.Counter(candidate_line) == collections.Counter(x):
                    #     print('not equal')
                    #     my_string = ' '.join(x)
                    #     my_string2 = ' '.join(candidate_line)

                    #     print(text_with_punctuation2.find(my_string))
                    #     number = cleaned_text2.find(my_string2)

                    #     if number < lowest_number:
                    #         lowest_number = number
                    #         my_string_saved = my_string
                    #         my_string_saved2 = my_string2


                    if debugging: print('')
                    text_with_punctuation[begin] = '\\textbf{'+text_with_punctuation[begin]
                    text_with_punctuation[end] = text_with_punctuation[end]+'}'
                except:
                    print('FAILED A CANDIDATE. PLEASE INVESTIGATE: ', candidate_line)
                    continue
                
            # print(lowest_number)
            # print(my_string_saved)
            # print(my_string_saved2)




            # print(counter)            
            text_with_punctuation = ' '.join(text_with_punctuation)
            print(text_with_punctuation)

            candidates_file.close()

            exit(0)

        candidates_file = './prose/' + prose_text + '_candidates.txt'
        if do_candidates:
            print('Creating hexameter candidates from given prose text')
            possible_hexameter_list = self.create_hexameter_list_from_window(all_text, self.min_syllables_hexameter, self.max_syllables_hexameter)     
            
            self.write_file(candidates_file, possible_hexameter_list)
        else:
            possible_hexameter_list = open(candidates_file, 'r')



        if do_lstm:
            # Let the LSTM think about what is acceptable as an hexameter
            # print('LSTM: creating candidate list')
            possible_hexameter_list = self.detect_hexameter_lstm(possible_hexameter_list)
            print(possible_hexameter_list, len(possible_hexameter_list))    
            # exit(0)        
        else:
            # Convert the syllabified list to Anceps & Pedecerto readable
            new_possible_hexameter_list = []
            for line in possible_hexameter_list:
                new_possible_hexameter_list.append(self.convert_split_hexameter_to_string(line))
            possible_hexameter_list = set(new_possible_hexameter_list)

        # exit(0)

        if do_anceps:
            # Let Anceps think about what is acceptable as an hexameter
            print('Anceps: creating candidate list')
            possible_hexameter_list = self.detect_hexameter_anceps(possible_hexameter_list)
            # print(possible_hexameter_list, len(possible_hexameter_list))

            anceps_candidates_file = prose_text + '_anceps_candidates'
            self.write_file(anceps_candidates_file, possible_hexameter_list)
            
            # with open('anceps_candidates.txt', 'w') as f:
            #     for candidate in possible_hexameter_list:
            #         f.write("%s\n" % candidate)
            # f.close()

        if do_pedecerto:
            # First, run all the Anceps candidates through Pedecerto
            print('Pedecerto: creating candidate list')
            possible_hexameter_list = self.detect_hexameter_pedecerto(possible_hexameter_list)
            
            # Next, interpret what Pedecerto put out
            from bs4 import BeautifulSoup
            pedecerto_candidates = []
            
            with open('/home/luukie/Downloads/scansion.xml') as fh:
                # Use beautiful soup to process the xml
                soupedEntry = BeautifulSoup(fh,"xml")
                # Clean the lines (done by MQDQ)
                soupedEntry = util.clean_extra(soupedEntry('line'))

                # text_sequence_label_list = []

                # for line in range(len(soupedEntry)):
                for line in range(len(soupedEntry)):
                    complete_line = []
                    for w in soupedEntry[line]("word"):
                        complete_line.append(w.text)

                    complete_line = ' '.join(complete_line)
                    pedecerto_candidates.append(complete_line)   

            pedecerto_candidates_file = prose_text + '_pedecerto_candidates'
            self.write_file(pedecerto_candidates_file, pedecerto_candidates)

    def read_file(self, filename):
        f = open(filename, "r")
        text = f.read()
        f.close()
        return text

    def write_file(self, filename, given_list):
        path = './prose/' + filename + '.txt'
        with open(path, 'w') as f:
            for item in given_list:
                f.write("%s\n" % item)
        f.close()        

    def find_sub_list(self, sl,l):
        """Function to find the location of a sublist within a list

        Args:
            sl (list): sublist to be found
            l (list): to find the sublist in

        Returns:
            ints: begin and end index of the substring within the list
        """        
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                return ind,ind+sll-1

    # def find_sub_list(self, sl,l):
    #     results=[]
    #     sll=len(sl)
    #     for ind in (i for i,e in enumerate(l) if e==sl[0]):
    #         if l[ind:ind+sll]==sl:
    #             results.append((ind,ind+sll-1))

    #     return results


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
        
        for i in result['text']:
            if result['text'][i]['method'] != 'failed':
                anceps_candidates.append(result['text'][i]['verse'])
        
        return anceps_candidates

    def detect_hexameter_pedecerto(self, hexameter_candidates):
        return hexameter_candidates

    def convert_split_hexameter_to_string(self, split_hexameter):
        original_line = ''.join([str(elem) for elem in split_hexameter])
        original_line = original_line.replace('-',' ')
        # if there are any multiple whitespaces, merge these
        original_line = ' '.join(original_line.split())
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
        for idx, word in enumerate(text.split()):
            syllabified = syl.syllabify(word)
            syllabified.append('-')
            syllabified_text += syllabified

        return syllabified_text

    def clean_prose_text(self, text):
        # Strip the text of non alphanumerical characters, lower it and merge whitespaces   
        text = text.translate({ord(c): None for c in "*,.:?![]';()~<>"})
        text = text.translate({ord(c): None for c in '0123456789'})
        # Delete everything that is completely uppercase. Usually titles.
        text = text.split()
        text = [item for item in text if not item.isupper()]
        text = [word.lower() for word in text]

        text = ' '.join(text)
        return text

    def check_word_boundaries(self, possible_hexameter_line):
        # An hexameter only works on word boundaries. this function checks that        
        return (possible_hexameter_line[0] == self.SPACE_CHAR and possible_hexameter_line[-1] == self.SPACE_CHAR)

    def calculate_hexameter_length(self, given_hexameter):
        return len([value for value in given_hexameter if value != '-'])

if __name__ == "__main__":
    prose = ProsePoetry(FLAGS)   
        
