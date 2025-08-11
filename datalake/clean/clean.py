import string 
from unidecode import unidecode
import unicodedata

allowed_characters_for_syllable = set(string.ascii_lowercase + '-')

def clean_dataset(dataset: list) -> list:
    """
    Cleans the given dataset
    """
    checked_lines: list = []
    cleaned_lines: list = []
    
    # First clean the lines 
    for line_object in dataset:
        # Write the cleaned lines to the cache
        cleaned_line = _clean_line(line_object)
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    # Reject lines (for example, those containing Greek)
    for line_object in cleaned_lines:
        if _check_if_line_is_acceptable(line_object['line']):
            checked_lines.append(line_object)


    return checked_lines

def _clean_line(line_object: dict)-> dict:
    # Check each syllable and clean it
    for obj in line_object['line']:
        if 'syllable' in obj:
            # If a line contains Greek, we will not use it.
            if _contains_greek(obj['syllable']):
                return {}
            obj['syllable'] = _remove_diacritics(obj['syllable'])
            obj['syllable'] = obj['syllable'].lower()
            # Replace any occurences of j by i
            obj['syllable'] = obj['syllable'].replace('j', 'i')
        if 'word' in obj:
            obj['word'] = obj['word'].lower()
            obj['word'] = _remove_diacritics(obj['word'])
            # Replace any occurences of j by i
            obj['word'] = obj['word'].replace('j', 'i')

    return line_object

def _remove_diacritics(s: str) -> str:
    return unidecode(s)

def _check_if_line_is_acceptable(line: list) -> bool:
    """
    Checks whether a line is acceptable to us.
    """
    for obj in line:
        if 'syllable' in obj:
            syllable: str = obj['syllable']
            length: str = obj['length']
            # We only accept lines that have alpha characters. So no Greek or punctuation
            if not all(c in allowed_characters_for_syllable for c in syllable):
                return False
            # Disqualify lines that have corrupt labels
            if length == 'corrupt' or syllable == '':
                return False
    return True

def _contains_greek(text: str)-> bool:
    for char in text:
        try:
            if 'GREEK' in unicodedata.name(char):
                return True
        except ValueError:
            # Character has no name (e.g., control character)
            continue
    return False
