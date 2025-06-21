import config as conf
import utilities as util
import re

def run(actions: list) -> None:
    if "all" in actions:
        create_dataset_of_everything()

    if "meter" in actions:
        per_meter()

    if "dactylic" in actions:
        create_dactylic() 

def create_dactylic() -> None:
    """
    Creates one dactylic dataset from the hexameter, pentameter and elegy datasets
    """
    cache: dict = { "lines": [] }
    # Get a list of files from all sources
    file_list: list = []
    file_list += [f"{conf.DATASETS_PATH}/per_meter/" + s for s in util.create_files_list(f"{conf.DATASETS_PATH}/per_meter", 'json')] 

    for file in file_list:
        meter: str = file.split('/')[-1].split('.')[0]
        print(f"processing {meter}")
        dactylic_meter: list = ['hexameter', 'pentameter', 'elegy']
        if meter in dactylic_meter:
            lines: dict = util.read_json(file)['lines']
            cache["lines"] += lines

        util.write_json(cache, f"{conf.DATASETS_PATH}/dactylic/dactylic.json")

def create_dataset_of_everything() -> None:
    # Get a list of files from all sources
    file_list: list = []
    file_list += [conf.CLEAN_PATH + '/' + s for s in util.create_files_list(conf.CLEAN_PATH, 'json')] 

    cache: dict = { "lines": [] }
    for file in file_list:
        print(f'processing {file}')

        lines: dict = util.read_json(file)['lines']
        cache["lines"] += lines

    util.write_json(cache, f"{conf.DATASETS_PATH}/all.json")

def per_meter() -> None:
    # Get a list of files from all sources
    file_list: list = []
    file_list += [conf.CLEAN_PATH + '/' + s for s in util.create_files_list(conf.CLEAN_PATH, 'json')] 

    # Find the types of meter in the datasets
    types_of_meter: dict = {} 
    for file in file_list:
        print(f'finding meters in {file}')
        # Open the file and read the lines property from the json
        lines: dict = util.read_json(file)['lines']
        for line in lines:
            if line['meter'] not in types_of_meter:
                types_of_meter[line['meter']] = 1
            else:
                types_of_meter[line['meter']] += 1

    # types_of_meter = {k: v for k, v in sorted(types_of_meter.items(), key=lambda item: item[1])}
    meters_to_process: list = []
    for meter in types_of_meter:
        if types_of_meter[meter] > 1000:
            meters_to_process.append(meter)

    # Now, for each meter, create a dataset for our lstm
    for meter in meters_to_process:
        cache: dict = { "lines": [] }
        for file in file_list:
            print(f'finding {meter} in {file}')

            lines: dict = util.read_json(file)['lines']
            filtered_list: list = [d for d in lines if d.get("meter", None) == meter]
            cache["lines"] += filtered_list

        util.write_json(cache, f"{conf.DATASETS_PATH}/per_meter/{meter}.json")
