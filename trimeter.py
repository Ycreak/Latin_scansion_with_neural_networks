import json
import re 
import argparse

import syllabifier as syl
import pickle


class Anceps():

    def __init__(self, FLAGS):

        if FLAGS.testing:
            with open('./pickle/sequence_labels/SEN-all.pickle', 'rb') as f:
                test = pickle.load(f)
                print(len(test))
                print(test[:2])

        if FLAGS.latin_library:
            self.convert_latin_library_text()

        if FLAGS.create_syllable_file:

            path = './anceps/full_scansions/'
            text_list = self.create_files_list(path, 'json')

            for anceps_text in text_list:
                file = path + anceps_text

                # Opening JSON file
                f = open(file)
                data = json.load(f)
                text = data['text']

                sequence_label_list = []

                for line in text:
                    # only do lines that are scanned by anceps and that are trimeter
                    if text[line]['method'] != 'automatic':
                        continue
                    if text[line]['meter'] != 'trimeter':
                        continue
                                
                    sentence_tupled = []

                    # Retrieve the verse in plain text and put it in a split list
                    current_verse = text[line]['scansion']
                    current_verse = current_verse.translate({ord(c): None for c in '()[]^_*'})
                    current_verse = current_verse.lower().split()

                    # Retrieve the scansions in a split list
                    current_line = text[line]['scansion'].strip()
                    current_line = current_line.split()
                
                    # Now retrieve the scansions and put them in a separate list
                    scansion_list = []
                    for syllable in current_line:
                        syllable_encoding = ''
                        for char in syllable:
                            if char == '*' or char == '^' or char == '(' or char == '[' or char == '_':
                                syllable_encoding += char
                        scansion_list.append(syllable_encoding)

                    # Now, split both the syllables and scansion
                    for idx, word in enumerate(current_verse):
                        word = re.sub(r'\W+', '', word)
                        syllabified = syl.syllabify(word)
                        # Get the scansions that belong to the current word
                        scansioned = list(scansion_list[idx])
                        # Convert the labels
                        scansioned = ['anceps' if x=='*' else 'long' if x=='_' else 'long' if x=='[' else 'short' if x=='^' else 'elision' if x=='(' else 'unk' for x in scansioned]
                        # Merge the syllabified words and the scansions into a sequence label list
                        new_tuple = self.merge(syllabified, scansioned)
                        # Also, add a space after each word
                        new_tuple.append(('-','space'))
                        # Do this for every word
                        sentence_tupled += new_tuple
                    

                    for item in sentence_tupled:
                        if item[1] == 'unk':
                            print('break')
                    sequence_label_list.append(sentence_tupled[:-1]) # We dont need the last space


                pickle_path = './pickle/sequence_labels/' + anceps_text.split('.')[0] + '.pickle'

                with open(pickle_path, 'wb') as f:
                    pickle.dump(sequence_label_list, f)

                f.close()
            
            # print(len(sequence_label_list))

            # linenumber = 1
            # for line in sequence_label_list:
            #     for syllable, label in line:
            #         print(linenumber, syllable, label)

            #     linenumber += 1
            
            # import pickle


        

    def merge(self, list1, list2):
        """Merge two lists into one -> [(l1_a, l2_a), (l1_b, l2_b)]

        Args:
            list1 (list): [description]
            list2 (list): [description]

        Returns:
            list: spliced together
        """    
        return list(zip(list1, list2)) 

    def create_files_list(self, path, substring):
        """Creates a list of files to be processed

        Args:
            path (string): folder to be searched
            substring (string): substring of files to be searched

        Returns:
            list: list with files to be searched
        """
        import os
        
        list = []

        for file in os.listdir(path):
            if file.find(substring) != -1:
                list.append(file)    

        return list

    def convert_latin_library_text(self):
        path = './trimeter/text/latinlibrary/'

        text_list = self.create_files_list(path, 'txt')


        for text in text_list:
            new_file = './trimeter/text/' + text
            g = open(new_file, 'w')
            file = path + text

            f = open(file, "r")
            for line in f.readlines():
                # Clean line from non-alpha numeric characters
                line = line.translate({ord(c): None for c in '(),.!?—;:*'})
                # Remove everything between brackets
                line = re.sub(r'\[[^)]*\]', '', line)
                # Remove numbers
                line = ''.join([i for i in line if not i.isdigit()])
                # Strip whitespaces
                line = line.strip().lower()
                line = line + '\n'

                g.write(line)

                print(line)
                # exit(0)
            g.close()
            f.close()
        # print(text_list)




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latin_library", action="store_true", help="specify whether to create the model: if not specified, we load from disk")
    p.add_argument("--sequence_label", action="store_true", help="specify whether to save the model: if not specified, we do not save")
    p.add_argument("--train_model", action="store_true", help="specify whether to train a FLAIR model")
    p.add_argument("--create_corpus", action="store_true", help="specify whether to create the corpus for FLAIR")
    p.add_argument("--create_syllable_file", action="store_true", help="specify whether to create a file consisting of syllables to train word vectors on")
    p.add_argument("--test_model", action="store_true", help="specify whether to test the FLAIR model")
    p.add_argument("--testing", action="store_true", help="specify whether to run the train/test split LSTM experiment")
    p.add_argument("--single_line", action="store_true", help="specify whether to predict a single line")

    p.add_argument("--verbose", action="store_true", help="specify whether to run the code in verbose mode")
    p.add_argument('--epochs', default=10, type=int, help='number of epochs')

    p.add_argument('--language_model', default='none', type=str, help='name of LM to train')


    FLAGS = p.parse_args()    
    
    my_anceps = Anceps(FLAGS)