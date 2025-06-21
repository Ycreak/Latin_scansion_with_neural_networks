import config as conf
import models.lstm.utilities as lstm_util
import utilities as util
import pandas as pd

def make_hexameter_heatmap(lstm, model) -> None:
    column_names = ["predictee", "label", "score"]
    df = pd.DataFrame(columns = column_names)

    file_list: list = []
    file_list += [conf.LSTM_SEQUENCE_LABELS_PATH + '/' + s for s in util.create_files_list(conf.LSTM_SEQUENCE_LABELS_PATH, 'json')] 

    for file in file_list:
        meter: str = file.split('/')[-1].split('.')[0]
        # if meter == 'hexameter':
            # continue

        print(f"predicting {meter}")

        sequence_labels_test_text: dict = util.read_json(file)['lines']

        # sequence_labels_test_text = util.read_json(f"{conf.LSTM_SEQUENCE_LABELS_PATH}/scazon.json")['lines']
        X_test, y_test = lstm.create_X_y_sets(sequence_labels_test_text, lstm.word2idx, lstm.label2idx, lstm.max_sentence_length)
        # classification_report = lstm.create_classification_report(model, X_test, y_test, False)
        # print(classification_report)

        classification_report = lstm.create_classification_report(model, X_test, y_test, True)
        score_long = float(round(classification_report['0']['f1-score'],2))
        score_short = float(round(classification_report['1']['f1-score'],2))
        score_elision = float(round(classification_report['2']['f1-score'],2))

        new_lines: list = [
            {'predictee': meter, 'label': 'long', 'score': score_long},
            {'predictee': meter, 'label': 'short', 'score': score_short},
            {'predictee': meter, 'label': 'elision', 'score': score_elision},
        ]
        df = pd.concat([df, pd.DataFrame(new_lines)], ignore_index=True)

    # pivot the dataframe to be usable with the seaborn heatmap
    heatmap_data = pd.pivot_table(df, values='score', index=['predictee'], columns='label')
    lstm_util.create_heatmap(dataframe = heatmap_data,
                    xlabel = 'label',
                    ylabel = 'predictee',
                    title = 'Confusion matrix -- hexameter predict',
                    filename = 'confusionmatrix_elegiac_short.png')
