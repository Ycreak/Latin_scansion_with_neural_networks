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

    def create_syllable_word_character_model(
        self, dataset: LSTMDataset
    ) -> tf.keras.models.Model:
        # These are padded, so all the same size. But to be sure, we calculate for each feature
        syllable_input_length = len(dataset.list_with_poetry_objects[0]["syllables"])
        word_input_length = len(dataset.list_with_poetry_objects[0]["words"])
        character_input_length = len(dataset.list_with_poetry_objects[0]["characters"])

        # Create input layers for both syllable and word tensors
        syllable_input_layer = tf.keras.layers.Input(
            shape=(syllable_input_length,), name="syllable_input"
        )
        word_input_layer = tf.keras.layers.Input(
            shape=(word_input_length,), name="word_input"
        )

        # character_encoding_dim = len(dataset.character_encoder.classes_)
        character_encoding_dim = len(
            dataset.list_with_poetry_objects[0]["characters"][0]
        )
        character_input_layer = tf.keras.layers.Input(
            shape=(character_input_length, character_encoding_dim),
            name="character_input",
        )

        # Embedding layers for both inputs
        syllable_embedding = tf.keras.layers.Embedding(
            input_dim=len(dataset.syllable_encoder.classes_),
            output_dim=self.embedding_output_dim,
            input_length=syllable_input_length,
        )(syllable_input_layer)

        word_embedding = tf.keras.layers.Embedding(
            input_dim=len(dataset.word_encoder.classes_),
            output_dim=self.embedding_output_dim,
            input_length=word_input_length,
        )(word_input_layer)

        # Add the character embeddings as a time distributed layer. We need this, as a TimeDistributed Dense layer can help in reducing
        # dimensionality or transforming these high dimensional one-hot encoded vectors into a more useful representation for the model.
        character_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.embedding_output_dim)
        )(character_input_layer)
        # Concatenate the embeddings
        concatenated = tf.keras.layers.concatenate(
            [syllable_embedding, word_embedding, character_dense]
        )

        # LSTM layer
        lstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=self.lstm_layer_units,
                return_sequences=True,
                recurrent_dropout=0.1,
            )
        )(concatenated)

        # Dense layer
        output_layer = tf.keras.layers.Dense(
            len(dataset.label_encoder.classes_), activation="softmax"
        )(lstm_layer)

        # Create the model
        model = tf.keras.models.Model(
            inputs=[syllable_input_layer, word_input_layer, character_input_layer],
            outputs=output_layer,
        )

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        return model

if __name__ == "__main__":
    lstm = TFLSTM()
    lstm.run()
