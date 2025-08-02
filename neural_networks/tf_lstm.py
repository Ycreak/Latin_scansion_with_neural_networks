from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
import polars as pl

from sklearn.model_selection import train_test_split
import numpy as np

from neural_networks.lstm_dataset import LSTMDataset

import torch

import tensorflow as tf

class TFLSTM:
    """
    This class creates the TensorFlow LSTM.
    """

    def __init__(self) -> None:
        # Tweak the LSTM model parameters
        self.embedding_output_dim: int = 25
        self.lstm_layer_units: int = 10 # 10
        self.LEARNING_RATE: float = 0.001

    def run(self) -> None:
        # Read the dataframe
        print('Reading parquet file.')
        df = pl.read_parquet('datalake/bucket/enriched/poetry/poetry_dataframe.parquet')
        df = df.filter(pl.col("meter") == "hexameter")

        dataset: LSTMDataset = LSTMDataset()
        dataset.run(df)

        # Create the model based on the dataset we created.
        model = self.create_model(dataset)

        # Make train and test datasets
        X_train, X_val, y_train, y_val = train_test_split(
            dataset.padded_syllable_tensors,
            dataset.padded_label_tensors,
            test_size=0.2,
            random_state=42
        )

        # Then train using X_train and y_train
        history = model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=5,
            validation_data=(X_val, y_val),
            verbose=True
        )

        # Predict probabilities
        y_pred_probs = model.predict(X_val)

        # Get predicted classes
        y_pred = np.argmax(y_pred_probs, axis=-1)

        y_true_flat = y_val.flatten()
        y_pred_flat = y_pred.flatten()

        classification_report2 = classification_report(
            y_true_flat,
            y_pred_flat,
            target_names=dataset.label_encoder.classes_
        )

    def create_model(self, dataset: LSTMDataset) -> tf.keras.models.Model:
        """
        Creates and returns the LSTM model
        :param dataset (LSTMDataset) from which to create the model 
        :return tf.keras.models.Model
        """
        # Initiate the model structure
        input_layer = tf.keras.layers.Input(shape=(len(dataset.padded_syllable_tensors[0]),))

        model = tf.keras.layers.Embedding(
            input_dim = len(dataset.syllable_encoder.classes_), 
            output_dim = self.embedding_output_dim, 
            input_length = len(dataset.padded_syllable_tensors[0])
        )(input_layer)

        model = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units = self.lstm_layer_units, 
                return_sequences = True, 
                recurrent_dropout = 0.1
            )
        )(model)

        model = tf.keras.layers.Dense(len(dataset.label_encoder.classes_), activation='softmax')(model)

        model = tf.keras.models.Model(input_layer, model)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate = self.LEARNING_RATE
            ),
            # Loss function to minimize
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            # List of metrics to monitor
            metrics=['accuracy'],
        )

        return model

if __name__ == "__main__":
    lstm = TFLSTM()
    lstm.run()
