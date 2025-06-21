import lsnn.utilities as util
from lsnn.clean.clean import clean_dataset

class Hypotactic:
    def run(self, source_path: str, destination_path: str) -> None:
        """
        Process the given poems: retrieves the author, meter and syllables from the given XML.
        Saves it to a dictionary with each entry having author, meter and a lines property. In
        the lines property, per syllable, the syllable itself, its length and its parent word
        is provided.
        """
        hypotactic_files: list = util.create_files_list(source_path, 'json')

        for file in hypotactic_files:
            file_name: str = file.split('.')[0]
            cleaned_dataset = clean_dataset(f"{source_path}/{file}")
            util.write_json(cleaned_dataset, f"{destination_path}/{file_name}.json")
