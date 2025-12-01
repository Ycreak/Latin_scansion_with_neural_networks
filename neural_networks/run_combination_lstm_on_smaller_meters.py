from sklearn.metrics import classification_report
import os
import polars as pl
from collections import defaultdict
from datetime import datetime

import numpy as np

import datalake.utilities as util

from sklearn.model_selection import train_test_split

from neural_networks.lstm_dataset import LSTMDataset
from neural_networks.common_lstm_tools import (
    generate_y_pred_true,
    get_candidate_meters_from_df,
)
from neural_networks.tf_lstm import TFLSTM


class Experiment:
    """
    This class creates the TensorFlow LSTM.
    """

    def __init__(self) -> None:
        self.EPOCHS: int = 3
        self.BATCH_SIZE: int = 32

    def run(self, save_model: bool = False) -> None:
        """
        This experiment trains lstms on the bigger datasets (dactylic, trochaic and ia6) and tries to scan the
        smaller datasets like scazon, hendecasyllables and senarii.
        """
        # Read the dataframe
        print("Reading parquet file.")
        df = pl.read_parquet("datalake/bucket/enriched/poetry/poetry_dataframe.parquet")
        df = df.filter(pl.col("meter") != "elegy") # TODO: elegy seems faulty. filter it away for now

        # Filter dataframe where we have at least 1000 lines
        candidate_meters_with_length = get_candidate_meters_from_df(df, 1000)

        # Filter the dataframe on only these meters
        candidate_meters = [obj["meter"] for obj in candidate_meters_with_length]
        df_with_candidate_meters = df.filter(pl.col("meter").is_in(candidate_meters))

        # We can train on datasets with 10k lines or more, and we will test on the smaller sets.
        training_meters = [
            obj["meter"] for obj in candidate_meters_with_length if obj["lines"] > 10000
        ]
        testing_meters = [
            obj["meter"]
            for obj in candidate_meters_with_length
            if obj["lines"] <= 10000
        ]

        # Create an LSTM dataset over the entire dataframe. We do this in order to be able to train on hexameter and test
        # on another meter, as the encoding needs to be the same for both datasets.
        dataset: LSTMDataset = LSTMDataset()
        dataset.run(df_with_candidate_meters)

        # We will train on all meters that have a sufficient set of lines
        train_meter_lines = [
            d
            for d in dataset.list_with_poetry_objects
            if d.get("meter") in training_meters
        ]

        # Get the tensors from the dataset we created
        syllable_tensors = np.array([d["syllables"] for d in train_meter_lines])
        word_tensors = np.array([d["words"] for d in train_meter_lines])
        label_tensors = np.array([d["labels"] for d in train_meter_lines])
        character_tensors = np.array([d["characters"] for d in train_meter_lines])

        # Split the tensors such that we have a validation dataset for training the model.
        indices = np.arange(len(syllable_tensors))
        train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)

        print("Creating training and validation sets.")
        syllable_train = np.array([syllable_tensors[i] for i in train_idx])
        syllable_test = np.array([syllable_tensors[i] for i in test_idx])

        word_train = np.array([word_tensors[i] for i in train_idx])
        word_test = np.array([word_tensors[i] for i in test_idx])

        label_train = np.array([label_tensors[i] for i in train_idx])
        label_test = np.array([label_tensors[i] for i in test_idx])

        character_train = np.array([character_tensors[i] for i in train_idx])
        character_test = np.array([character_tensors[i] for i in test_idx])

        # Create the model based on the dataset we created.
        tf_lstm: TFLSTM = TFLSTM()

        # Create a new model for training.
        print("Creating the model.")
        model = tf_lstm.create_model_with_word_tensors(dataset)

        # Then train using the features we have created
        print("Fitting the model.")
        model.fit(
            [syllable_train, word_train, character_train],
            label_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=([syllable_test, word_test, character_test], label_test),
            verbose=True,
        )

        if save_model:
            model.save('neural_networks/models/combination_lstm_on_smaller_meters.keras')

        # Create a dict where we can save our results to
        results = defaultdict(list)

        # For each testing meter, see how well our model can predict this meter.
        for meter in testing_meters:
            print(f"Now testing on {meter}")
            test_meter_lines = [
                d for d in dataset.list_with_poetry_objects if d.get("meter") == meter
            ]
            syllable_tensors = np.array([d["syllables"] for d in test_meter_lines])
            word_tensors = np.array([d["words"] for d in test_meter_lines])
            label_tensors = np.array([d["labels"] for d in test_meter_lines])
            character_tensors = np.array([d["characters"] for d in test_meter_lines])

            # Now test the model on our smaller datasets.
            y_pred, y_true = generate_y_pred_true(
                model,
                [syllable_tensors, word_tensors, character_tensors],
                label_tensors,
            )

            report: dict = classification_report(
                y_true,
                y_pred,
                target_names=dataset.label_encoder.classes_,
                output_dict=True,
            )

            # Save the classification report to the results dictionary for later use.
            results[meter].append(report)

        # Write the experiment to a timestamped folder for later use.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        file_path: str = (
            "neural_networks/experiments/run_combination_lstm_on_smaller_meters"
        )
        os.makedirs(file_path, exist_ok=True)
        util.write_json(results, f"{file_path}/{timestamp}.json")


if __name__ == "__main__":
    experiment = Experiment()
    experiment.run(save_model=True)
