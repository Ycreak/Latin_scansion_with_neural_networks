from sklearn.metrics import classification_report
import os 
import polars as pl
from sklearn.model_selection import KFold
from collections import defaultdict
from datetime import datetime

import numpy as np

import datalake.utilities as util

from neural_networks.lstm_dataset import LSTMDataset
from neural_networks.tf_lstm import TFLSTM
from neural_networks.common_lstm_tools import generate_y_pred_true, get_candidate_meters_from_df

class Experiment:
    """
    This class creates the TensorFlow LSTM.
    """

    def __init__(self) -> None:
        self.FOLDS = 2
        self.EPOCHS: int = 5
        self.BATCH_SIZE: int = 32 

    def run(self) -> None:
        """
        This experiment trains and tests an lstm on a single type of meter. For this, we take all meters in our dataframe
        and only train/test when we have more than 10k lines, as this is a minimum dataset size to properly train an LSTM.
        """
        # Read the dataframe
        print('Reading parquet file.')
        df = pl.read_parquet('datalake/bucket/enriched/poetry/poetry_dataframe.parquet')

        # Filter away those meters with too little lines to make a proper dataset.
        candidate_meters_with_length = get_candidate_meters_from_df(df, 1000)
        candidate_meters = [obj['meter'] for obj in candidate_meters_with_length]
        df_with_candidate_meters = df.filter(pl.col("meter").is_in(candidate_meters))
       
        # Create an LSTM dataset over the entire dataframe. We do this in order to be able to train on hexameter and test
        # on another meter, as the encoding needs to be the same for both datasets.
        dataset: LSTMDataset = LSTMDataset()
        dataset.run(df_with_candidate_meters)

        # Create a dict where we can save our results to
        results = defaultdict(list)

        for meter in candidate_meters:
            print(f"Now processing {meter}")
            # candidate_meter_df = dataset.dataframe.filter(pl.col("meter") == meter)
            candidate_meter_lines = [d for d in  dataset.list_with_poetry_objects if d.get("meter") == meter]

            # Get the tensors from the dataset we created
            syllable_tensors = np.array([d["syllables"] for d in candidate_meter_lines]) 
            # syllable_tensors = np.array(candidate_meter_df.select("syllable_tensors").to_series().to_list())
            word_tensors = np.array([d["words"] for d in candidate_meter_lines])
            # word_tensors = np.array(candidate_meter_df.select("word_tensors").to_series().to_list())
            label_tensors = np.array([d["labels"] for d in candidate_meter_lines])
            # label_tensors = np.array(candidate_meter_df.select("label_tensors").to_series().to_list())
            character_tensors = np.array([d["characters"] for d in candidate_meter_lines])
            # Create the model based on the dataset we created.
            tf_lstm: TFLSTM = TFLSTM()
            
            # K-Fold Cross-Validation
            kf = KFold(n_splits=self.FOLDS, shuffle=True, random_state=42)
            for fold, (train_idx, test_idx) in enumerate(kf.split(syllable_tensors)):
                print(f"\n=== Fold {fold + 1} ===")

                # Create a new model for training.
                model = tf_lstm.create_model_with_word_tensors(dataset)

                syllable_train, syllable_test = syllable_tensors[train_idx], syllable_tensors[test_idx]
                word_train, word_test = word_tensors[train_idx], word_tensors[test_idx]
                label_train, label_test = label_tensors[train_idx], label_tensors[test_idx]

                # Then train using X_train and y_train
                model.fit(
                    [syllable_train, word_train],
                    label_train,
                    batch_size=self.BATCH_SIZE,
                    epochs=self.EPOCHS,
                    validation_data=([syllable_test, word_test], label_test),
                    verbose=True
                )

                y_pred, y_true = generate_y_pred_true(
                    model,
                    [syllable_test, word_test],
                    label_test
                ) 
            
                report: dict = classification_report(
                    y_true,
                    y_pred,
                    target_names=dataset.label_encoder.classes_,
                    output_dict=True
                )

                # Save the classification report to the results dictionary for later use.
                results[meter].append(report)
                
        # Write the experiment to a timestamped folder for later use.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        file_path: str = "neural_networks/experiments/run_lstm_on_each_meter_type"
        os.makedirs(file_path, exist_ok=True)
        util.write_json(results, f"{file_path}/{timestamp}.json")

if __name__ == "__main__":
    experiment = Experiment()
    experiment.run()
