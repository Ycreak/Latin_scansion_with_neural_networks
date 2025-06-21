# import config as conf
import utilities as util
import re
from mqdq import rhyme as mqdq_rhyme
from mqdq.cltk_hax.syllabifier import Syllabifier
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import dataclasses

@dataclasses.dataclass
class Syllable:
    syllable: str 
    word: str
    length: str

class Pedecerto:
    def run(self, source_path: str, destination_path: str) -> None:
        """This class parses the Pedecerto XML into a syllable label list which can be used for
        training models. The parsed files are stored in a pickle in the corresponding folder (see config).

        NB: corrupt lines or lines containing words that could not be syllabified will be stripped!
        NB: the XML files are stripped of their headers, leaving the body to be processed.
        """
        # Add all entries to process to a list
        xml_files = util.create_files_list(conf.PEDECERTO_XML_PATH, 'xml')
        cache_list: list[str] = util.create_files_list(conf.PEDECERTO_XML_PATH, 'json')
        cache_list = [name.replace('.json', '') for name in cache_list]

        # Process all entries added to the list
        for entry in xml_files:
            if entry.replace('.xml', '') not in cache_list:
                self.convert_pedecerto_xml_to_dictionary(entry)

    def convert_pedecerto_xml_to_dictionary(self, xml_name: str):
        cache: dict = { "lines": [] }
        with open(conf.PEDECERTO_XML_PATH+ xml_name) as fh:
            text_name = xml_name.split('.')[0]
            print(f"processing {text_name}")
            # Use beautiful soup to process the xml
            soupedEntry = BeautifulSoup(fh,"xml")
            # Retrieve the title and author from the xml file
            text_title = str(soupedEntry.title.string)
            author = str(soupedEntry.author.string)
            # Clean the lines (done by MQDQ)
            soupedEntry = self._clean(soupedEntry('line'))

            for line in range(len(soupedEntry)):
                line_list: list = []

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
                    line_list, success = self._process_line(soupedEntry[line])
                    if success:
                        # Only add the line if no errors occurred.
                        cache['lines'].append({
                            'author': author,
                            'meter': self._get_meter(soupedEntry[line]['meter']),
                            'line': line_list
                            })
                else:
                    continue # interestingly, some pedecerto xml files use "metre" instead of "meter"
            
        util.write_json(cache, f"{conf.PEDECERTO_DICTIONARY_PATH}/{text_name}.json")


    def _clean(self, ll):

        """Remove all corrupt lines from a set of bs4 <line>s

        Args:
            ll (list of bs4 <line>): Lines to clean

        Returns:
            (list of bs4 <line>): The lines, with the corrupt ones removed.
        """

        return [
            l
            for l in ll
            if l.has_attr("pattern")
            and l["pattern"] != "corrupt"
            and l["pattern"] != "not scanned"
        ]

    def _get_meter(self, string: str) -> str:
        """
        Returns a string of the meter given the character pedecerto uses as denominator
        """
        if string == "H":
            return 'hexameter'
        elif string == "P":
            return "pentameter"
        else:
            return "unknown"

    def _process_line(self, given_line):
        """Processes a given XML pedecerto line. Puts syllable and length in a dataframe.

        Args:
            given_line (xml): pedecerto xml encoding of a line of poetry

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
                # line_sequence_label_list.append((current_syllable, length))
                syllable_object: Syllable = Syllable(
                        syllable=current_syllable,
                        word=w.text,
                        length=length
                    )
                line_sequence_label_list.append(dataclasses.asdict(syllable_object))
            # At the end of the word, we add a space, unless it is the last word of the given line
            line_sequence_label_list.append({'-':'space'}) 
        # we dont need a space after the last word, so drop the last row of the list
        return line_sequence_label_list[:-1], True # Boolean to denote success of syllabification

