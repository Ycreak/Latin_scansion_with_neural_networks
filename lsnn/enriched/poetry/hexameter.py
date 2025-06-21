from sources.hypotactic.harvester import Harvester
from sources.hypotactic.soup_poem import PoemSoup
from sources.hypotactic.soup_text import TextSoup
import sources.hypotactic.utilities as util
import config as conf

# Get a list of all cached webpages
webpage_cache_list: list = util.get_files_list(conf.HYPOTACTIC_WEBPAGE_PATH)
soup_cache_list: list[str] = util.get_files_list(conf.HYPOTACTIC_DICTIONARY_PATH)
soup_cache_list = [name.replace('.json', '') for name in soup_cache_list]

# List to save all lines from all texts to
cache: dict = { "lines": [] }

class Hexameter:
    def run(self) -> None:
        self.process_poems()
        self.process_prose()

    def process_poems(self) -> None:
        # Harvest all poems
        for poem in poems:
            print(f'doing soup on {poem}')
            page = util.read_pickle(f'{conf.HYPOTACTIC_WEBPAGE_PATH}/{poem}')
            my_soup = PoemSoup(page, poem)

            # Combine all soups into one big dictionary
            # print(f"combining cache with {poem}")
            # poem = util.read_json(f'{conf.HYPOTACTIC_SOUP_PATH}/{poem}.json')
            # cache['lines'] += poem['lines']
            # util.write_json(cache, f"{conf.HYPOTACTIC_DICTIONARY_FILE}")

    def process_prose(self) -> None:
        # Harvest all prose
        for text in prose:
            print(f'doing soup on {text}')
            page = util.read_pickle(f'{conf.HYPOTACTIC_WEBPAGE_PATH}/{text}')
            my_soup = TextSoup(page, text)

            # Combine all soups into one big dictionary
            # print(f"combining cache with {text}")
            # text = util.read_json(f'{conf.HYPOTACTIC_SOUP_PATH}/{text}.json')
            # cache['lines'] += text['lines']
            # util.write_json(cache, f"{conf.HYPOTACTIC_DICTIONARY_PATH}")
