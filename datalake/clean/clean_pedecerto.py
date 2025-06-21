import datalake.utilities as util
from datalake.clean.clean import clean_dataset

class Pedecerto:
    def run(self, source_path: str, destination_path: str) -> None:
        """
        Clean lines from Pedecerto
        """
        pedecerto_files: list = util.create_files_list(source_path, 'json')

        for file in pedecerto_files:
            file_name: str = file.split('.')[0]
            cleaned_dataset = clean_dataset(f"{source_path}/{file}")
            util.write_json(cleaned_dataset, f"{destination_path}/{file_name}.json")
