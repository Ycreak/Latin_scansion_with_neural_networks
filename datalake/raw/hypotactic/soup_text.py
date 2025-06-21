from bs4 import BeautifulSoup
import utilities as util
import sources.hypotactic.utilities as hypotactic_util 
import config as conf
import dataclasses
import time 

@dataclasses.dataclass
class Syllable:
    syllable: str 
    word: str
    length: str

class TextSoup:
    def __init__(self, page, slug: str):
        self.slug: str = slug
        soup = BeautifulSoup(page, 'html.parser')
        lines = soup.find_all('div', class_='line')

        cache: dict = { "lines": [] }
        self.process(lines, cache)

    def process(self, lines, cache: dict) -> None:
        """
        Process the given lines: retrieves the author, meter and syllables from the given XML.
        Saves it to a dictionary with each entry having author, meter and a lines property. In
        the lines property, per syllable, the syllable itself, its length and its parent word
        is provided.
        """
        for line in lines:
            meter: str = line['data-metre'] if line and 'data-metre' in line.attrs else 'Unknown'
            # use the slug as the author for now
            author: str = self.slug

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

            cache['lines'].append({
                'author': author,
                'meter': meter,
                'line': line_list
                })

        util.write_json(cache, f"{conf.HYPOTACTIC_DICTIONARY_PATH}/{self.slug}.json")

