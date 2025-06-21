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
                    
class Pedecerto_parser:
    """This class parses the Pedecerto XML into a syllable label list which can be used for
    training models. The parsed files are stored in a pickle in the corresponding folder (see config).

    NB: corrupt lines or lines containing words that could not be syllabified will be stripped!
    NB: the XML files are stripped of their headers, leaving the body to be processed.
    """  

    def convert_pedecerto_xml_to_syllable_sequence_files(self,
        input_folder: str,
        output_folder: str
        ):
        # Add all entries to process to a list
        entries = util.create_files_list(input_folder, 'xml')
        # Process all entries added to the list

        for entry in entries:
            with open(input_folder + entry) as fh:
                # for each text, an individual dataframe will be created and saved as pickle
                text_name = entry.split('.')[0]
                pickle_name = text_name + '.pickle'
                # Use beautiful soup to process the xml
                soupedEntry = BeautifulSoup(fh,"xml")
                # Retrieve the title and author from the xml file
                text_title = str(soupedEntry.title.string)
                author = str(soupedEntry.author.string)
                # Clean the lines (done by MQDQ)
                soupedEntry = util.clean(soupedEntry('line'))

                text_sequence_label_list = []
                # for line in range(len(soupedEntry)):
                for line in Bar('Processing {0}, {1}'.format(author, text_title)).iter(range(len(soupedEntry))):
                    try:
                        book_title = int(soupedEntry[line].parent.get('title'))
                    except:
                        book_title = 1 # if we cant extract a title (if there isnt any like ovid ibis) -> just 1.
                    # Process the entry. It will append the line to the df
                    if not soupedEntry[line]['name'].isdigit(): # We only want lines that are certain
                        continue
                    if soupedEntry[line]['pattern'] == 'not scanned': # These lines we also skip
                        continue
                    if soupedEntry[line]['meter'] == "H" or soupedEntry[line]['meter'] == "P": # We only want hexameters or pentameters
                        line_sequence_label_list, success = self.process_line(soupedEntry[line])
                        if success:
                            # Only add the line if no errors occurred.
                            text_sequence_label_list.append(line_sequence_label_list)
                    else:
                        continue # interestingly, some pedecerto xml files use "metre" instead of "meter"

            util.pickle_write(output_folder, pickle_name, text_sequence_label_list)

    def process_line(self, given_line):
        """Processes a given XML pedecerto line. Puts syllable and length in a dataframe.

        Args:
            given_line (xml): pedecerto xml encoding of a line of poetry
            df (dataframe): to store the data in
            book_title (str): title of the current book (Book 1)

        Returns:
            list: with syllable label tuples for the entire given sentence
            boolean: whether the process of creating the sentence was successful
        """      
        # Create a list in which we will save the sequence_labels for every syllable
        line_sequence_label_list = []
        # Parse every word and add its features
        for w in given_line("word"):
            # Now for every word, syllabify it first
            try:
                word_syllable_list = mqdq_rhyme._syllabify_word(w).syls
            except:
            # Could not syllabify. Notify the calling function to not add this line to the set
                return line_sequence_label_list, False
            # And get its scansion
            scansion = w["sy"]
            # Check how many syllables we have according to pedecerto
            split_scansion = [scansion[i:i+2] for i in range(0, len(scansion), 2)] # per two characters
            # We use this to detect elision      
            number_of_scansions = len(split_scansion)
            for i in range(len(word_syllable_list)):
                # Now we loop through the syllable list of said word and extract features
                current_syllable = word_syllable_list[i].lower()
                # If we still have scansions available
                if number_of_scansions > 0:
                    feet_pos = split_scansion[i][1]
                    # Interpret length based on pedecerto encoding (could be done much quicker)
                    if feet_pos.isupper():
                        length = 'long'
                    elif feet_pos.islower():
                        length = 'short'
                # No scansions available? Elision.
                else:
                    length = 'elision' # Luckily elision happens only at the end of a word
                # Keep track of performed operations
                number_of_scansions -= 1
                # Append to dataframe
                line_sequence_label_list.append((current_syllable, length))
            # At the end of the word, we add a space, unless it is the last word of the given line
            line_sequence_label_list.append(('-', 'space')) 
        # we dont need a space after the last word, so drop the last row of the list
        return line_sequence_label_list[:-1], True # Boolean to denote success of syllabification

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--pedecerto", action="store_true", help="run the pedecerto parser")
#     p.add_argument("--anceps", action="store_true", help="run the anceps parser")
#     p.add_argument("--combine_datasets", action="store_true", help="combines the datasets in the sequence labels folder")
#     FLAGS = p.parse_args()    
    
#     if FLAGS.anceps:
#         Anceps_parser().convert_anceps_json_to_syllable_sequence_files(
#             input_folder = conf.ANCEPS_SCANSION_FOLDER,
#             output_folder = conf.SEQUENCE_LABELS_FOLDER
#         )

#     if FLAGS.pedecerto:
#         Pedecerto_parser().convert_pedecerto_xml_to_syllable_sequence_files(
#             input_folder = conf.PEDECERTO_SCANSION_FOLDER,
#             output_folder = conf.SEQUENCE_LABELS_FOLDER
#         )

#     if FLAGS.combine_datasets:
#         # combine all sequence label files in the sequence_labels folder
#         combined_folder = conf.SEQUENCE_LABELS_FOLDER
#         entries = util.create_files_list(combined_folder, 'pickle')
#         if(entries):
#             util.combine_sequence_label_lists(entries, 'combined.pickle', conf.SEQUENCE_LABELS_FOLDER, add_extension=False) 
