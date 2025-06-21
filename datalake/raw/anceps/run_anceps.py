# Python imports
import json
import re 
import argparse
import pickle

# Library imports
from bs4 import BeautifulSoup
from progress.bar import Bar
from mqdq import rhyme as mqdq_rhyme
from mqdq.cltk_hax.syllabifier import Syllabifier

# Class imports
from lsnn import utilities as util

class Anceps_parser():
    """This class parses the Anceps JSON files into a syllable label list which can be used for
    training models. The parsed files are stored in a pickle in the corresponding folder (see config).
    """  
    def convert_anceps_json_to_syllable_sequence_files(self,
        input_folder: str,
        output_folder: str
        ):
        
        # path = conf.ANCEPS_SCANSION_FOLDER
        text_list = util.create_files_list(input_folder, 'json')
        syllabify_word = Syllabifier().syllabify

        for anceps_text in text_list:
            print('Working on {0}'.format(anceps_text))
            file = input_folder + anceps_text

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
                    syllabified = syllabify_word(word)
                    # Get the scansions that belong to the current word
                    scansioned = list(scansion_list[idx])
                    # Convert the labels
                    scansioned = ['anceps' if x=='*' else 'long' if x=='_' else 'long' if x=='[' else 'short' if x=='^' else 'elision' if x=='(' else 'unk' for x in scansioned]
                    # Merge the syllabified words and the scansions into a sequence label list
                    new_tuple = util.merge_lists(syllabified, scansioned)
                    # Also, add a space after each word
                    new_tuple.append(('-','space'))
                    # Do this for every word
                    sentence_tupled += new_tuple
                
                for item in sentence_tupled:
                    if item[1] == 'unk':
                        print('break')
                sequence_label_list.append(sentence_tupled[:-1]) # We dont need the last space

            pickle_path = output_folder + anceps_text.split('.')[0] + '.pickle'

            with open(pickle_path, 'wb') as f:
                pickle.dump(sequence_label_list, f)

            f.close()

