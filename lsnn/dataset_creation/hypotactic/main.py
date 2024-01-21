from harvester import Harvester
from soup import Soup

from pickler import Pickler

class Mainer:
    def __init__(self, reharvest, resoup):
        self.harvester = Harvester()
        self.pickler = Pickler()

        self.resoup: bool = resoup
        self.reharvest: bool = reharvest

        if self.resoup or self.reharvest:
        # Read all the texts
            f = open("pages.txt", "r")
            texts = f.read().splitlines()
            f.close()

            for text in texts:
                print(f'processing {text}')
                page_source = self.harvester.run(text, reharvest)
                my_soup = Soup(page_source)

        result = self.pickler.read('lines')

        print(result)

        for key in result:
            if len(result[key]) > 100:
                print(key, len(result[key]))
                for line in result[key]:
                    for syllable, label in line:
                        print(line)


if __name__ == "__main__":
    main = Mainer(
            reharvest = False,
            resoup = True
            )






