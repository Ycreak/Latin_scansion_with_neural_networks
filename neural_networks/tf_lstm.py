from neural_networks.lstm_dataset import LSTMDataset
import numpy as np


import tensorflow as tf

class TFLSTM:
    """
    This class creates the TensorFlow LSTM.
    """

    def __init__(self) -> None:
        # Tweak the LSTM model parameters
        self.embedding_output_dim: int = 25
        self.lstm_layer_units: int = 10
        self.LEARNING_RATE: float = 0.001

    def create_model_with_word_tensors(self, dataset: LSTMDataset) -> tf.keras.models.Model:
        # Get the input length for syllable and word tensors
        # len([d["syllables"] for d in dataset.list_with_poetry_objects])
        
        # These are padded, so all the same size. But to be sure, we calculate for each feature
        syllable_input_length = len(dataset.list_with_poetry_objects[0]['syllables'])
        word_input_length = len(dataset.list_with_poetry_objects[0]['words'])

        # Create input layers for both syllable and word tensors
        syllable_input_layer = tf.keras.layers.Input(shape=(syllable_input_length,), name='syllable_input')
        word_input_layer = tf.keras.layers.Input(shape=(word_input_length,), name='word_input')

        # Embedding layers for both inputs
        syllable_embedding = tf.keras.layers.Embedding(
            input_dim=len(dataset.syllable_encoder.classes_),
            output_dim=self.embedding_output_dim,
            input_length=syllable_input_length
        )(syllable_input_layer)

        word_embedding = tf.keras.layers.Embedding(
            input_dim=len(dataset.word_encoder.classes_),
            output_dim=self.embedding_output_dim,
            input_length=word_input_length
        )(word_input_layer)

        # Concatenate the embeddings
        concatenated = tf.keras.layers.concatenate([syllable_embedding, word_embedding])

        # LSTM layer
        lstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=self.lstm_layer_units,
                return_sequences=True,
                recurrent_dropout=0.1
            )
        )(concatenated)

        # Dense layer
        output_layer = tf.keras.layers.Dense(len(dataset.label_encoder.classes_), activation='softmax')(lstm_layer)

        # Create the model
        model = tf.keras.models.Model(inputs=[syllable_input_layer, word_input_layer], outputs=output_layer)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )

        return model


    def create_model(self, dataset: LSTMDataset) -> tf.keras.models.Model:
        """
        Creates and returns the LSTM model
        :param dataset (LSTMDataset) from which to create the model 
        :return tf.keras.models.Model
        """
        # As the input length is the same for all lines of poetry, we can pick the length of the first syllable tensor.
        input_length = len(dataset.dataframe.select("syllable_tensors").to_series().to_list()[0])
        # Initiate the model structure
        input_layer = tf.keras.layers.Input(shape=(input_length,))
        
        model = tf.keras.layers.Embedding(
            input_dim = len(dataset.syllable_encoder.classes_), 
            output_dim = self.embedding_output_dim, 
            input_length = input_length 
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
