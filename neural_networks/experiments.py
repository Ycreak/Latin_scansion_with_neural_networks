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


import tensorflow as tf

class Experiments:
    """
    This class creates the TensorFlow LSTM.
    """

    def __init__(self) -> None:
        self.FOLDS = 2
        self.EPOCHS: int = 20
        self.BATCH_SIZE: int = 32 

    def run_lstm_on_each_meter_type(self) -> None:
        """
        This experiment trains and tests an lstm on a single type of meter. For this, we take all meters in our dataframe
        and only train/test when we have more than 10k lines, as this is a minimum dataset size to properly train an LSTM.
        """
        # Read the dataframe
        print('Reading parquet file.')
        df = pl.read_parquet('datalake/bucket/enriched/poetry/poetry_dataframe.parquet')

        # Find the number of lines we have per meter type
        meters_with_counts = (
            df.group_by("meter")
            .agg(pl.col("line_number").n_unique().alias("number_of_lines"))
            .sort("number_of_lines", descending=True)
        )

        # Filter dataframe where we have at least 1000 lines
        candidate_meters = [d['meter'] for d in meters_with_counts.to_dicts() if d["number_of_lines"] > 1000]
        df_with_candidate_meters = df.filter(pl.col("meter").is_in(candidate_meters))
       
        # occurences_per_label = self._get_number_of_occurences_per_label(df)
        # print(occurences_per_label)

        # Create an LSTM dataset over the entire dataframe. We do this in order to be able to train on hexameter and test
        # on another meter, as the encoding needs to be the same for both datasets.
        dataset: LSTMDataset = LSTMDataset()
        dataset.run(df_with_candidate_meters)

        # Create a dict where we can save our results to
        results = defaultdict(list)

        for meter in candidate_meters:
            print(f"Now processing {meter}")
            candidate_meter_df = dataset.dataframe.filter(pl.col("meter") == meter)
            # Get the tensors from the dataframe
            syllable_tensors = np.array(candidate_meter_df.select("syllable_tensors").to_series().to_list())
            label_tensors = np.array(candidate_meter_df.select("label_tensors").to_series().to_list())
            # Create the model based on the dataset we created.
            tf_lstm: TFLSTM = TFLSTM()
            
            # K-Fold Cross-Validation
            kf = KFold(n_splits=self.FOLDS, shuffle=True, random_state=42)
            for fold, (train_idx, test_idx) in enumerate(kf.split(syllable_tensors)):
                print(f"\n=== Fold {fold + 1} ===")

                # Create a new model for training.
                model = tf_lstm.create_model(dataset)

                syllable_train = syllable_tensors[train_idx]
                syllable_test = syllable_tensors[test_idx]

                label_train = label_tensors[train_idx]
                label_test = label_tensors[test_idx]

                # Then train using X_train and y_train
                model.fit(
                    syllable_train,
                    label_train,
                    batch_size=self.BATCH_SIZE,
                    epochs=self.EPOCHS,
                    validation_data=(syllable_test, label_test),
                    verbose=True
                )

                y_pred, y_true = self.generate_y_pred_true(model, syllable_test, label_test)
            
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
        file_path: str = "neural_networks/experiments/run_lstm_on_each_meter_type/"
        os.makedirs(file_path, exist_ok=True)
        util.write_json(results, f"{file_path}/{timestamp}.json")


    def generate_y_pred_true(self, model: tf.keras.models.Model, test_features: list, test_labels: list) -> tuple[list, list]:
        """
        Generates a y_prediction and a y_true dataset given the model. Can be used to run a classification report or predict a sentence.
        :param model (tf.keras.models.Model)
        :param test_features (list) with features which to predict
        :param test_labels (list) with labels to predict
        :return tuple[list, list] with predictions and true labels
        """
        # Predict probabilities
        y_pred_probs = model.predict(test_features)

        # Get predicted classes
        y_pred = np.argmax(y_pred_probs, axis=-1)

        y_true_flat = test_labels.flatten()
        y_pred_flat = y_pred.flatten()

        return y_pred_flat, y_true_flat

    def _get_number_of_occurences_per_label(self, df: pl.DataFrame) -> dict:
        # Get the number of occurences per label for each of our candidate meters.
        filtered_df = df.filter(pl.col("meter"))#.is_in(candidate_meters))
        counts_per_meter_and_label = (
            filtered_df.group_by(["meter", "label"])
            .agg(pl.count().alias("count"))
        )

        counts_dict = {}
        for meter, group in counts_per_meter_and_label.group_by("meter"):
            counts_dict[meter] = dict(zip(group["label"], group["count"]))
        # Show the result
        return counts_dict


if __name__ == "__main__":
    experiments = Experiments()
    experiments.run_lstm_on_each_meter_type()
