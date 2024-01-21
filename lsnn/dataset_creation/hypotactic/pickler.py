import pickle
import os

class Pickler:
    
    def __init__(self):
        pass

    def save(self, slug: str, content: any):
        file: str = f"./pickle/{slug}.pickle"

        with open(file, 'wb') as f:
            pickle.dump(content, f)

        f.close()

    def read(self, slug: str):
        print(f'Loading pickled {slug}.')
        file: str = f"./pickle/{slug}.pickle"

        with open(file, 'rb') as f:
            return pickle.load(f)

    def exists(self, slug: str) -> bool:
        return os.path.isfile(f"./pickle/{slug}.pickle")

