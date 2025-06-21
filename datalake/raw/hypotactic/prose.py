import datalake.utilities as util

from bs4 import BeautifulSoup
import datalake.utilities as util
import datalake.raw.hypotactic.utilities as hypotactic_util 
import dataclasses

@dataclasses.dataclass
class Syllable:
    syllable: str 
    word: str
    length: str

class Prose:
    def run(self, source_path: str, destination_path: str) -> None:
        """
        Process the given lines: retrieves the author, meter and syllables from the given XML.
        Saves it to a dictionary with each entry having author, meter and a lines property. In
        the lines property, per syllable, the syllable itself, its length and its parent word
        is provided.
        """
        hypotactic_prose_files: list = util.create_files_list(source_path, 'prose')
        
        for text in hypotactic_prose_files:
            file_name = text.split('.')[0]
            print(f'doing soup on {text}')

            page = util.read_pickle(f'{source_path}/{text}')
            soup = BeautifulSoup(page, 'html.parser')
            lines = soup.find_all('div', class_='line')

            # List to save all lines in the text to.
            all_lines: list = []

            for line in lines:
                meter: str = line['data-metre'] if line and 'data-metre' in line.attrs else 'Unknown'
                # use the filename as the author for now
                author: str = file_name

                line_list: list = []
                words = line.find_all('span', class_='word')

                for word in words:
                    syllables = word.find_all('span', class_='syll')

                    # Reconstruct the full word from the syllables
                    full_word: str = ''
                    for syllable in syllables:
                        clean_syllable: str = hypotactic_util.format_string(syllable.get_text())
                        full_word += syllable.text

                    for syllable in syllables:
                        clean_syllable: str = hypotactic_util.format_string(syllable.get_text())
                        length = hypotactic_util.retrieve_length(syllable['class'])

                        syllable_object: Syllable = Syllable(
                                syllable=clean_syllable,
                                word=hypotactic_util.format_string(full_word),
                                length=length
                            )
                        line_list.append(dataclasses.asdict(syllable_object))
                    line_list.append({"-":"space"})

                # Remove the last entry from the line, which is a superfluous space
                line_list = line_list[:-1] 

                all_lines.append({
                    'author': author,
                    'meter': meter,
                    'line': line_list
                    })

            util.write_json(all_lines, f"{destination_path}/{file_name}.json")

