import config as conf
import utilities as util
import re
from unidecode import unidecode

def run(actions: list) -> None:
    if "run" in actions:
        clean_dataset()

def clean_dataset() -> None:
    # Get a list of files from all sources
    file_list: list = []
    file_list += [conf.HYPOTACTIC_DICTIONARY_PATH + '/' + s for s in util.create_files_list(conf.HYPOTACTIC_DICTIONARY_PATH, 'json')] 
    file_list += [conf.PEDECERTO_DICTIONARY_PATH + '/' + s for s in util.create_files_list(conf.PEDECERTO_DICTIONARY_PATH, 'json')] 
    

    for file in file_list:
        file_name: str = file.split('/')[-1].split('.')[0]
        print(f"processing {file_name}")
        cache: dict = { "lines": [] }
        # Open the file and read the lines property from the json
        lines: dict = util.read_json(file)['lines']
        checked_lines: list = []

        # Reject lines (for example, those containing Greek)
        for line_object in lines:
            if _check_line(line_object['line']):
                checked_lines.append(line_object)

        # Next clean those lines we did not reject on a syllable level
        for line_object in checked_lines:
            # Check each syllable and clean it
            for obj in line_object['line']:
                if 'syllable' in obj:
                    # Remove all diacritics
                    # if any(ord(char) > 127 for char in obj['syllable']):
                        # temp = True
                    obj['syllable'] = _remove_diacritics(obj['syllable'])
            # Write the cleaned lines to the cache
            cache['lines'].append(line_object)

        util.write_json(cache, f"{conf.CLEAN_PATH}/{file_name}.json")

def _remove_diacritics(s: str) -> str:
    return unidecode(s)

def _check_line(line: list) -> bool:
    """
    Checks whether a line is acceptable to us.
    """
    for obj in line:
        if 'syllable' in obj:
            syllable: str = obj['syllable']
            length: str = obj['length']
            # We only accept lines that have alpha characters. So no Greek or punctuation
            if re.match(r'^[^a-zA-Z]*$', syllable):
                return False
            # Disqualify lines that have corrupt labels
            if length == 'corrupt':
                return False
    return True
