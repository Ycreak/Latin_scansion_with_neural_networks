from bs4 import BeautifulSoup
from pickler import Pickler

import string

class Soup:
    def __init__(self, page):
        self.pickler = Pickler()
        if self.pickler.exists('lines'):
            self.meters = self.pickler.read('lines')
        else:
            self.meters = {}

        soup = BeautifulSoup(page, 'html.parser')
        # First, check whether we are dealing with poems or drama
        poems = soup.find_all('div', class_='poem')
        drama = soup.find_all('div', class_='persona')

        if (len(poems) > 0):
            # Process poems
            for poem in poems:
                meter = poem['data-metre'] if poem and 'data-metre' in poem.attrs else None
                lines = poem.find_all('div', class_='line')
                for line in lines:
                    self.process_line(line, meter)

        elif (len(drama) > 0):
            # Process drama
            lines = soup.find_all('div', class_='line')
            for line in lines:
                meter = line['data-metre'] if line and 'data-metre' in line.attrs else None
                self.process_line(line, meter)

        else:
          raise Exception("Unknown data type.") 
        
        self.pickler.save('lines', self.meters)

    def process_line(self, line: list, meter: str):
        accepted = False
        line_list: list = []
        words = line.find_all('span', class_='word')

        labels = ['long', 'short', 'elided', 'space']

        for word in words:
            syllables = word.find_all('span', class_='syll')

            for syllable in syllables:
                # The length is encoded in the class, which looks as follows:
                # class="syll short hiatus", or class="syll resolution short". 
                # To get the lenght, we have to search the class elements for one of our defined labels
                # if the length does not exist, we have a line with corruption, which we will skip.
                syllable_class_element = syllable['class']

                for element in syllable_class_element:
                    if element in labels:
                        accepted = True
                        length = element
                        break

                # Length is not always defined. Fix this plz
                print(line)
                print(length)

                syllable = syllable.get_text().lower()
                syllable = syllable.translate(str.maketrans('', '', string.punctuation))
                line_list.append((syllable, length)) 

            line_list.append(('-', 'space'))

        if meter not in self.meters:
            self.meters[meter] = []
        if accepted:
            self.meters[meter].append(line_list[:-1]) # We dont need the last space

