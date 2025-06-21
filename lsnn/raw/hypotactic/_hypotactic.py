from lsnn.raw.hypotactic.soup_poem import PoemSoup
from lsnn.raw..hypotactic.soup_text import TextSoup
import utilities as util


class Hypotactic:
    def run(self, source_path: str, destination_path: str) -> None:
        self.source_path = source_path
        self.destination_path = destination_path

        self.process_poems()
        self.process_prose()

    def process_poems(self) -> None:
        # Harvest all poems
        for poem in poems:
            print(f'doing soup on {poem}')
            page = util.read_pickle(f'{self.source_path}/{poem}.html')
            my_soup = PoemSoup(page, poem)

    def process_prose(self) -> None:
        # Harvest all prose
        for text in prose:
            print(f'doing soup on {text}')
            page = util.read_pickle(f'{self.source_path}/{text}.html')
            my_soup = TextSoup(page, text)
