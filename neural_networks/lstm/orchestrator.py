import config as conf
from models.lstm.experiments import make_hexameter_heatmap
import utilities as util

from models.lstm.lstm import LSTM

def run(actions: list) -> None:
    if "dataset" in actions:
        _create_sequence_label_lists()

    if "train" in actions:
        lstm: LSTM = LSTM()
        model = lstm.create_model(
            num_epochs = 10,
            text = 'all.json', 
            save_model = True, 
            model_name = 'hexameter10'
        )
        # model = lstm.load_model('hexameter25')
        make_hexameter_heatmap(lstm, model)

        # classification_report = lstm.create_classification_report(model, X_test, y_test, True)
        # score_long = float(round(classification_report['0']['f1-score'],2))
        # score_short = float(round(classification_report['1']['f1-score'],2))
        # score_elision = float(round(classification_report['2']['f1-score'],2))

    if "test" in actions:
        lstm: LSTM = LSTM()
        model = lstm.load_model('hexameter25')
        make_hexameter_heatmap(lstm, model)

def _create_sequence_label_lists() -> None:
    # Read all datasets and convert them to LSTM readable sequence label lists
    meter_files_path: str = f"{conf.DATASETS_PATH}"
    file_list: list = []
    file_list += [meter_files_path + '/' + s for s in util.create_files_list(meter_files_path, 'json')] 

    for file in file_list:
        meter: str = file.split('/')[-1].split('.')[0]
        print(f"processing {meter}")

        output_dict = {"lines":[]}
        sequence_labels_list: list = []


        lines: dict = util.read_json(file)['lines']
        for line_dict in lines:
            sequence_labels_line_list: list = []
            line: list = line_dict['line'] 
            for item in line:
                if 'syllable' in item:
                   sequence_labels_line_list.append((item['syllable'], item['length'])) 
                if 'space' in item:
                   sequence_labels_line_list.append(('-', 'space')) 

            sequence_labels_list.append(sequence_labels_line_list)
        output_dict['lines'] = sequence_labels_list
        util.write_json(output_dict, f"{conf.LSTM_SEQUENCE_LABELS_PATH}/{meter}.json")


