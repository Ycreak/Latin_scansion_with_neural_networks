import datalake.utilities as util
from datalake.clean.clean import clean_dataset


class Pedecerto:
    def run(self, source_path: str, destination_path: str) -> None:
        """
        Clean lines from Pedecerto
        """
        pedecerto_files: list = util.create_files_list(source_path, "json")

        for file in pedecerto_files:
            file_name: str = file.split(".")[0]
            print(f"Processing {file}")
            dataset = util.read_json(f"{source_path}/{file}")
            cleaned_dataset = clean_dataset(dataset)
            util.write_json(cleaned_dataset, f"{destination_path}/{file_name}.json")

    # def _clean_extra(ll):
    # """Remove all corrupt lines from a set of bs4 <line>s, but also those that are uncertain

    # Args:
    # ll (list of bs4 <line>): Lines to clean

    # Returns:
    # (list of bs4 <line>): The lines, with the corrupt ones removed.
    # """
    # temp = []

    # for line in lines:
    # if line.has_attr("feature"):
    # if line["feature"] != "spondaic":
    # temp.append(line)
    # else:
    # temp.append(line)

    # ll = temp

    # ll = [
    # l
    # for l in ll
    # if l.has_attr("pattern")
    # and l["pattern"] != "corrupt"
    # and l["pattern"] != "not scanned"
    # and l["pattern"] != "SSSS"
    # ]

    # return ll
