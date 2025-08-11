from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torch
import tensorflow as tf
import polars as pl
from tqdm import tqdm
import string

class LSTMDataset:
    """
    Creates an LSTM compatible dataset from the given dataframe.
    """
    def __init__(self) -> None:
        # Padding string used in the LSTM to make all lines the same length.
        self.PADDING = '@'

        # Allow the encoders to be later used by other components.
        self.syllable_encoder: LabelEncoder
        self.label_encoder: LabelEncoder
        self.word_encoder: LabelEncoder 
        self.character_encoder : LabelEncoder

        self.test_padded_syllable_tensors = [] 

        # Save all data into one nice dataframe
        self.dataframe: pl.DataFrame

    def run(self, df: pl.DataFrame) -> None:
        print('Building dataframe with one row per line of poetry.')
        poetry_line_per_row_df = (
            df
            .group_by("line_number")
            .agg([
                pl.col("syllable").alias("syllables"),
                pl.col("label").alias("labels"),
                pl.col("word").alias("words"),
                pl.col("meter").first().alias("meter")  
            ])
        )

        # poetry_line_per_row_df = poetry_line_per_row_df.sample(500)

        # Now we need all our features and labels in separate lists, including the padding string.
        all_syllables: list[str] = df['syllable'].to_list() + [self.PADDING]
        all_labels: list[str] = df['label'].to_list() + [self.PADDING]
        all_words: list[str] = df['word'].to_list() + [self.PADDING]
        all_characters: list[str] = list(string.ascii_lowercase) + ['-', self.PADDING] # add space and padding chars

        # An LSTM can only accept integers, so we use one-hot encoding to turn our strings into integers.
        # We do this for all our inputs and our labels. First, create the encoders themselves.
        print('Creating encoders.')
        self.syllable_encoder: LabelEncoder = LabelEncoder().fit(all_syllables)
        self.label_encoder: LabelEncoder = LabelEncoder().fit(all_labels)
        self.word_encoder: LabelEncoder = LabelEncoder().fit(all_words)
        self.character_encoder: LabelEncoder = LabelEncoder().fit(all_characters)

        # For every syllable we have, create a character tensor, which is a list with a tensor for each character in the syllable.
        all_unique_syllables = list(set(all_syllables))

        # Process character tensors for each syllable. We need to create per syllable a list with its
        # character tensors. So both 'a' and 'r' in 'ar' will get a tensor put in a list.
        character_tensor_per_syllable= defaultdict(dict)
        for syllable in all_unique_syllables:
            char_indices = self.character_encoder.transform(all_unique_syllables)
            char_tensor = tf.keras.utils.to_categorical(char_indices, num_classes=len(self.character_encoder.classes_))
            character_tensor_per_syllable[syllable] = char_tensor

        # FIXME: in the datalake, remove all lines that have non a-z characters

        # Example syllable
        syllable = "ar"

        # Convert characters to integers
        char_indices = self.character_encoder.transform(list(syllable))

        # One-hot encode the characters
        char_tensor = tf.keras.utils.to_categorical(char_indices, num_classes=len(self.character_encoder.classes_))

        print(char_tensor)

        # Now for each line of poetry, we must one-hot encode its syllables, words and labels.
        # So per line in our dataframe, use the encode to turn [ar, ma] into e.g. [12, 14].
        print('Preparing sequences.')
        # Create lookups to speedup the creation of tensors.  
        word_to_id = {i: idx for idx, i in enumerate(self.word_encoder.classes_)}
        syllable_to_id = {i: idx for idx, i in enumerate(self.syllable_encoder.classes_)}
        label_to_id = {i: idx for idx, i in enumerate(self.label_encoder.classes_)}
        character_to_id = {char: idx for idx, char in enumerate(all_characters)}

        # Create lists for all tensors. These we will later add to the dataframe
        syllable_tensors: list[torch.Tensor] = []
        word_tensors: list[torch.Tensor] = []
        label_tensors: list[torch.Tensor] = []
        character_tensors: list = []

        print('Number of iterations to do:', poetry_line_per_row_df.height)
        for row in tqdm(poetry_line_per_row_df.iter_rows(named=True)):
            # Use the lookup for each feature and the labels to one-hot encode our strings.
            ids = [word_to_id.get(word) for word in row['words']]
            word_tensor: torch.Tensor = torch.tensor(ids)
            word_tensors.append(word_tensor)

            ids = [syllable_to_id.get(syllable) for syllable in row['syllables']]
            syllable_tensor: torch.Tensor = torch.tensor(ids)
            syllable_tensors.append(syllable_tensor)

            ids = [label_to_id.get(label) for label in row['labels']]
            label_tensor: torch.Tensor = torch.tensor(ids)
            label_tensors.append(label_tensor)

        # Now we have for every line of poetry a list of one-hot encoded syllables, words and labels for that list.
        # For example, for the line [ar, ma] we have three lists: [12, 14], [55, 55], [0, 1] for syllables, words and labels.
        # An LSTM wants a lot of lines of the same length, so we use padding to make every line the same length.
        PADDING_INTEGER_SYLLABLE = self.syllable_encoder.transform(['PADDING'])[0]
        PADDING_INTEGER_WORD = self.word_encoder.transform(['PADDING'])[0]
        PADDING_INTEGER_LABEL = self.label_encoder.transform(['PADDING'])[0]

        padded_syllable_tensors = pad_sequence(syllable_tensors, batch_first=True, padding_value=PADDING_INTEGER_SYLLABLE)
        padded_word_tensors = pad_sequence(word_tensors, batch_first=True, padding_value=PADDING_INTEGER_WORD)
        padded_label_tensors = pad_sequence(label_tensors, batch_first=True, padding_value=PADDING_INTEGER_LABEL)
        
        self.test_padded_syllable_tensors = pad_sequence(syllable_tensors, batch_first=True, padding_value=PADDING_INTEGER_SYLLABLE)
        
        # Add the tensors as a new column to the DataFrame
        self.dataframe = poetry_line_per_row_df.with_columns(
            pl.Series("word_tensors", padded_word_tensors),
            pl.Series("syllable_tensors", padded_syllable_tensors),
            pl.Series("label_tensors", padded_label_tensors)
        )

