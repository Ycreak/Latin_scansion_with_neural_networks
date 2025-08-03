from neural_networks.lstm_dataset import LSTMDataset


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
