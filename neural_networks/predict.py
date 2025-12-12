import polars as pl
import numpy as np

from neural_networks.lstm_dataset import LSTMDataset
from neural_networks.common_lstm_tools import (
    get_candidate_meters_from_df,
)
from tensorflow import keras
import pickle 
from sklearn.preprocessing import LabelEncoder

class Predictor:
    """
    This class creates the TensorFlow LSTM.
    """

    def __init__(self) -> None:
        self.EPOCHS: int = 3
        self.BATCH_SIZE: int = 32

    def run(self, df_with_line_to_predict: pl.DataFrame) -> None:
        """
        This experiment trains lstms on the bigger datasets (dactylic, trochaic and ia6) and tries to scan the
        smaller datasets like scazon, hendecasyllables and senarii.
        """
        # Load the model
        model = keras.models.load_model('neural_networks/models/combination_lstm_on_smaller_meters.keras')

        with open('neural_networks/models/syllable_encoder.pickle', 'rb') as file:
            syllable_encoder: LabelEncoder = pickle.load(file)
        with open('neural_networks/models/word_encoder.pickle', 'rb') as file:
            word_encoder: LabelEncoder = pickle.load(file)
        with open('neural_networks/models/character_encoder.pickle', 'rb') as file:
            character_encoder: LabelEncoder = pickle.load(file)
        with open('neural_networks/models/label_encoder.pickle', 'rb') as file:
            label_encoder: LabelEncoder = pickle.load(file)

        print("Reading parquet file.")
        df = pl.read_parquet("datalake/bucket/enriched/poetry/poetry_dataframe.parquet")
        df = df.filter(pl.col("meter") != "elegy") # TODO: elegy seems faulty. filter it away for now

        print(df_with_line_to_predict)

        # Filter dataframe where we have at least 1000 lines
        candidate_meters_with_length = get_candidate_meters_from_df(df, 1000)
        # Filter the dataframe on only these meters
        candidate_meters = [obj["meter"] for obj in candidate_meters_with_length]
        df = df.filter(pl.col("meter").is_in(candidate_meters))
        
        # Test with the first line
        # df_with_line_to_predict= df.filter(pl.col("line_number")==1)
       
        # print(df_with_line_to_predict)
        # So here we are doing some pretty dumb stuff. We want to reuse the lstm_dataset.run function. For that, we need to provide a dataframe.
        # This dataframe should contain the lines we want to predict, the longest line (as we use this for padding) and the line with the longest
        # syllable, as we use this for padding the character tensors xD. This is so dumb, but i thought it was a nice shortcut, which it clearly
        # is no longer. Ah well...
        longest_line = (
            df
            .group_by("line_number")
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
            .head(1)
        )["line_number"][0]
        df_with_longest_line = df.filter(pl.col("line_number")==longest_line)
   
        # Find the line with the longest syllable
        unique_syllables: list[str] = df["syllable"].unique().to_list()
        longest_syllable: str = max(unique_syllables, key=len)

        line_number_of_longest_syllable: int = (
            df
            .filter(pl.col("syllable") == longest_syllable)
            .select("line_number")
            .to_series()
            .to_list()
        )[0]
        df_with_longest_syllable = df.filter(pl.col("line_number")==line_number_of_longest_syllable)

        df = pl.concat([df_with_longest_line, df_with_longest_syllable, df_with_line_to_predict], how="diagonal")

        # Create an lstm dataframe from our candidate lines
        dataset: LSTMDataset = LSTMDataset()
        dataset.run(df, syllable_encoder=syllable_encoder, word_encoder=word_encoder, character_encoder=character_encoder, label_encoder=label_encoder)

        # Remove from our test set the longest line and the one with the longest syllable
        # Only keep the lines we want to predict (which i saved in the 'meter' key)
        test_meter_lines = [obj for obj in  dataset.list_with_poetry_objects if obj['meter'] == 'predict']

        syllable_tensors = np.array([d["syllables"] for d in test_meter_lines])
        word_tensors = np.array([d["words"] for d in test_meter_lines])
        character_tensors = np.array([d["characters"] for d in test_meter_lines])

        # Predict probabilities using the model with multiple input features
        y_pred_probs = model.predict([syllable_tensors, word_tensors, character_tensors])
        # Get predicted classes
        y_pred = np.argmax(y_pred_probs, axis=-1)
        # Decode predicted classes to their original labels
        decoded_labels = dataset.label_encoder.inverse_transform(y_pred[0])
        decoded_syllables = dataset.syllable_encoder.inverse_transform(syllable_tensors[0])

        result = []
        # Per syllable, show per label the probability
        for probs in y_pred_probs[0]:
            syllable_probs = {label: float(prob) for label, prob in zip(dataset.label_encoder.classes_, probs)}
            result.append(syllable_probs)
        # Add said syllable to each object
        result = [
            {"syllable": syllable, **probs}
            for syllable, probs in zip(decoded_syllables, result)
        ]
        result = [
            {"prediction": syllable, **probs}
            for syllable, probs in zip(decoded_labels, result)
        ]
    
        return result

if __name__ == "__main__":
    experiment = Predictor()
    experiment.run(save_model=True)
