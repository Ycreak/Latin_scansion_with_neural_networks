import re

# CLTK related
import cltk
from pedecerto.rhyme import *

from bs4 import BeautifulSoup

import utilities as util

class Text_preprocessor:
    """ This class takes a pedecerto text and puts all words in a list, removing all non alpha 
    characters. Also does lowercasing. Most importantly, it divides all words in syllables via
    the code made available by the MQDQ project (https://github.com/bnagy/mqdq-parser)
    """
    def __init__(self, text):

        file_list = util.Create_files_list('./texts', 'xml')

        for file in file_list: 

            full_file = './texts/' + file

            word_list = []
            # TODO: find a good way to append multiple texts (for sake of word embeddings)
            with open(full_file) as fh:
                # Use beautiful soup to process the xml
                soupedEntry = BeautifulSoup(fh,"xml")
                # Take only line entries approved by pedecerto #FIXME: this might skew results
                soupedEntry = util.clean(soupedEntry('line'))
                # Create the word list by looping through each line of the text
                for line in range(len(soupedEntry)):
                    # Extend the word list with the syllables of each line
                    word_list.extend(self.Syllabify_line(soupedEntry[line]))

            # Clean the text (already partly done by pedecerto)
            word_list = self.Remove_numbers(word_list)
            word_list = self.Remove_element_from_list(word_list, '')
            word_list = self.Lowercase_list(word_list)

            # Store this character list for later retrieval
            self.character_list = word_list

    def Syllabify_line(self, givenLine):
        """Given a pedecerto XML line, all syllables are returned.

        Args:
            givenLine (xml): pedecerto xml encoded line of poetry

        Returns:
            list: of syllables in given line
        """        
        all_syllables = syllabify_line(givenLine)
        # Flatten list (hack)
        all_syllables = [item for sublist in all_syllables for item in sublist]

        return all_syllables

    def Lowercase_list(self, given_list):
        """Takes a list to lowercase all characters given

        Args:
            given_list (list): to lowercase

        Returns:
            list: lowercased list
        """        
        return list(map(lambda x: x.lower(), given_list))

    def Remove_element_from_list(self, given_list, element):
        """Removes given element from the given list

        Args:
            given_list (list): with element to remove
            element (string): of element to remove

        Returns:
            list: of items without the provided element
        """        
        return list(filter(lambda x: x != element, given_list))

    def Remove_numbers(self, list): 
        """ Removes all the numbers and punctuation marks from a given list.

        Args:
            list (list): we want to clean

        Returns:
            list: cleaned of numbers and punctuation
        """        
        pattern = '[0-9]|[^\w]'
        list = [re.sub(pattern, '', i) for i in list] 
        return list

    # # These functions may be of later use (perseus texts)
    # def Get_word_list(self, text):
    #     """Reads the given texts and returns its words in a list

    #     Args:
    #         text (string): of the cltk json file

    #     Returns:
    #         list: of words
    #     """    
    #     reader = get_corpus_reader( corpus_name = 'latin_text_perseus', language = 'latin')
    #     docs = list(reader.docs())
    #     reader._fileids = [text]
    #     words = list(reader.words())

    #     return words

    # # Strips the accents from the text for easier searching the text.
    # # Accepts string with accentuation, returns string without.
    # def strip_accents(self, s):
    #     return ''.join(c for c in unicodedata.normalize('NFD', s)
    #         if unicodedata.category(c) != 'Mn')

    # # Lemmatizes a given list. Returns a list with lemmatized words.
    # def lemmatizeList(self, list):
    #     tagger = POSTag('greek')

    #     lemmatizer = LemmaReplacer('greek')
    #     lemmWords = lemmatizer.lemmatize(list)

    #     # Remove Stopwords and numbers and lowercases all words.
    #     lemmWords = [w.lower() for w in lemmWords if not w in STOPS_LIST]
    #     lemmWords = removeNumbers(lemmWords)

    #     return lemmWords    