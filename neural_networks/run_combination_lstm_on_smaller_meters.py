from sklearn.metrics import classification_report
import os 
import polars as pl
from sklearn.model_selection import KFold
from collections import defaultdict
from datetime import datetime

import numpy as np

import datalake.utilities as util

from neural_networks.lstm_dataset import LSTMDataset
from neural_networks.common_lstm_tools import generate_y_pred_true, get_candidate_meters_from_df
from neural_networks.tf_lstm import TFLSTM


import tensorflow as tf

class Experiment:
    """
    This class creates the TensorFlow LSTM.
    """

    def __init__(self) -> None:
        self.FOLDS = 2
        self.EPOCHS: int = 1
        self.BATCH_SIZE: int = 32 

    def run(self) -> None:
        """
        This experiment trains lstms on the bigger datasets (dactylic, trochaic and ia6) and tries to scan the
        smaller datasets like scazon, hendecasyllables and senarii.
        """
        # Read the dataframe
        print('Reading parquet file.')
        df = pl.read_parquet('datalake/bucket/enriched/poetry/poetry_dataframe.parquet')

        # Filter dataframe where we have at least 1000 lines
        candidate_meters_with_length = get_candidate_meters_from_df(df, 1000)
        
        # Filter the dataframe on only these meters
        candidate_meters = [obj['meter'] for obj in candidate_meters_with_length]
        df_with_candidate_meters = df.filter(pl.col("meter").is_in(candidate_meters))
     
        # We can train on datasets with 10k lines or more, and we will test on the smaller sets.
        training_meters = [obj['meter'] for obj in candidate_meters_with_length if obj['lines'] > 10000]
        testing_meters = [obj['meter'] for obj in candidate_meters_with_length if obj['lines'] <= 10000]

        # Create an LSTM dataset over the entire dataframe. We do this in order to be able to train on hexameter and test
        # on another meter, as the encoding needs to be the same for both datasets.
        dataset: LSTMDataset = LSTMDataset()
        dataset.run(df_with_candidate_meters)

        # We will train on all meters that have a sufficient set of lines
        train_meter_df = dataset.dataframe.filter(pl.col("meter").is_in(training_meters))

        # Get the tensors from the dataframe
        train_syllable_tensors = np.array(train_meter_df.select("syllable_tensors").to_series().to_list())
        train_label_tensors = np.array(train_meter_df.select("label_tensors").to_series().to_list())
        
        # Create the model based on the dataset we created.
        tf_lstm: TFLSTM = TFLSTM()
        
        # Create a new model for training.
        model = tf_lstm.create_model(dataset)

        # Then train using X_train and y_train
        model.fit(
            train_syllable_tensors,
            train_label_tensors,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            verbose=True
        )

        # Create a dict where we can save our results to
        results = defaultdict(list)

        # For each testing meter, see how well our model can predict this meter.
        for meter in testing_meters:
            print(f"Now testing on {meter}")
            test_meter_df = dataset.dataframe.filter(pl.col("meter") == meter)
            test_syllable_tensors = np.array(test_meter_df.select("syllable_tensors").to_series().to_list())
            test_label_tensors = np.array(test_meter_df.select("label_tensors").to_series().to_list())

            # Now test the model on our smaller datasets.
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
        file_path: str = "neural_networks/experiments/run_combination_lstm_on_smaller_meters"
        os.makedirs(file_path, exist_ok=True)
        util.write_json(results, f"{file_path}/{timestamp}.json")


if __name__ == "__main__":
    experiment = Experiment()
    experiment.run()
