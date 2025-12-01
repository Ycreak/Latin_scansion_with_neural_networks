from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

import torch
import polars as pl
from tqdm import tqdm
import string


class LSTMDataset:
    """
    Creates an LSTM compatible dataset from the given dataframe.
    """

    def __init__(self) -> None:
        # Padding string used in the LSTM to make all lines the same length.
        self.PADDING = "@"

        # Allow the encoders to be later used by other components.
        self.syllable_encoder: LabelEncoder
        self.label_encoder: LabelEncoder
        self.word_encoder: LabelEncoder
        self.character_encoder: LabelEncoder

        # Save all data into one nice dataframe
        self.dataframe: pl.DataFrame
        self.list_with_poetry_objects: list = []

    def run(self, df: pl.DataFrame) -> None:
        print("Building dataframe with one row per line of poetry.")
        poetry_line_per_row_df = df.group_by("line_number").agg(
            [
                pl.col("syllable").alias("syllables"),
                pl.col("label").alias("labels"),
                pl.col("word").alias("words"),
                pl.col("meter").first().alias("meter"),
            ]
        )

        # poetry_line_per_row_df = poetry_line_per_row_df.sample(500)

        # Now we need all our features and labels in separate lists, including the padding string.
        all_syllables: list[str] = df["syllable"].to_list() + [self.PADDING]
        all_labels: list[str] = df["label"].to_list() + [self.PADDING]
        all_words: list[str] = df["word"].to_list() + [self.PADDING]
        all_characters: list[str] = list(string.ascii_lowercase) + [
            "-",
            self.PADDING,
        ]  # add space and padding chars

        # An LSTM can only accept integers, so we use one-hot encoding to turn our strings into integers.
        # We do this for all our inputs and our labels. First, create the encoders themselves.
        print("Creating encoders.")
        self.syllable_encoder: LabelEncoder = LabelEncoder().fit(all_syllables)
        self.label_encoder: LabelEncoder = LabelEncoder().fit(all_labels)
        self.word_encoder: LabelEncoder = LabelEncoder().fit(all_words)
        self.character_encoder: LabelEncoder = LabelEncoder().fit(all_characters)

        # For every syllable we have, create a character tensor, which is a list with a tensor for each character in the syllable.
        all_unique_syllables = list(set(all_syllables))

        # Process character tensors for each syllable. We need to create per syllable a list with its
        # character tensors. So both 'a' and 'r' in 'ar' will get a tensor put in a list. We also need to pad this,
        # such that each syllable has a list of character tensors that has the same length.
        character_tensors_per_syllable = defaultdict(list)
        max_syllable_length = max(len(ele) for ele in all_unique_syllables)

        for syllable in all_unique_syllables:
            char_tensor_for_current_syllable = []
            for char in syllable:
                # Create a one-hot encoded tensor for the character
                char_index = self.character_encoder.transform([char])[0]
                char_tensor_for_current_syllable.append(char_index)

            # So now we have for a single syllable a list with per character an integer as one-hot encoding. We need to pad this tensor to be as long
            # as the longest syllable in the dataset, such that every tensor has the same length. We ue the padding integer to pad each tensor.
            while len(char_tensor_for_current_syllable) < max_syllable_length:
                char_tensor_for_current_syllable.append(
                    self.character_encoder.transform([self.PADDING])[0]
                )

            # Store the padded sequence in the dictionary
            character_tensors_per_syllable[syllable] = char_tensor_for_current_syllable

        # Now for each line of poetry, we must one-hot encode its syllables, words and labels.
        # So per line in our dataframe, use the encode to turn [ar, ma] into e.g. [12, 14].
        print("Preparing sequences.")
        # Create lookups to speedup the creation of tensors.
        word_to_id = {i: idx for idx, i in enumerate(self.word_encoder.classes_)}
        syllable_to_id = {
            i: idx for idx, i in enumerate(self.syllable_encoder.classes_)
        }
        label_to_id = {i: idx for idx, i in enumerate(self.label_encoder.classes_)}

        # Create lists for all tensors. These we will later add to the dataframe
        syllable_tensors: list[torch.Tensor] = []
        word_tensors: list[torch.Tensor] = []
        label_tensors: list[torch.Tensor] = []
        character_tensors: list[list[torch.Tensor]] = []

        print("Number of iterations to do:", poetry_line_per_row_df.height)
        for row in tqdm(poetry_line_per_row_df.iter_rows(named=True)):
            # Use the lookup for each feature and the labels to one-hot encode our strings.
            ids = [word_to_id.get(word) for word in row["words"]]
            word_tensor: torch.Tensor = torch.tensor(ids)
            word_tensors.append(word_tensor)

            ids = [syllable_to_id.get(syllable) for syllable in row["syllables"]]
            syllable_tensor: torch.Tensor = torch.tensor(ids)
            syllable_tensors.append(syllable_tensor)

            ids = [label_to_id.get(label) for label in row["labels"]]
            label_tensor: torch.Tensor = torch.tensor(ids)
            label_tensors.append(label_tensor)

            char_tensors_for_row = [
                torch.tensor(character_tensors_per_syllable[syllable])
                for syllable in row["syllables"]
            ]
            character_tensors.append(char_tensors_for_row)

        # Now we have for every line of poetry a list of one-hot encoded syllables, words and labels for that list.
        # For example, for the line [ar, ma] we have three lists: [12, 14], [55, 55], [0, 1] for syllables, words and labels.
        # An LSTM wants a lot of lines of the same length, so we use padding to make every line the same length.
        print("Padding tensors.")
        PADDING_INTEGER_SYLLABLE = self.syllable_encoder.transform([self.PADDING])[0]
        PADDING_INTEGER_WORD = self.word_encoder.transform([self.PADDING])[0]
        PADDING_INTEGER_LABEL = self.label_encoder.transform([self.PADDING])[0]
        PADDING_TENSOR_CHARACTER = character_tensors_per_syllable[self.PADDING]

        padded_syllable_tensors = pad_sequence(
            syllable_tensors, batch_first=True, padding_value=PADDING_INTEGER_SYLLABLE
        )
        padded_word_tensors = pad_sequence(
            word_tensors, batch_first=True, padding_value=PADDING_INTEGER_WORD
        )
        padded_label_tensors = pad_sequence(
            label_tensors, batch_first=True, padding_value=PADDING_INTEGER_LABEL
        )

        # Now for every character tensor, we need to do some manual padding because I cannot figure out how to use the pad_sequence with nested lists.
        max_line_length: int = max(len(ele) for ele in label_tensors)
        # For each element in character_tensors, add the padding tensor as many times as to get to the max line length.
        padded_character_tensors = []
        for char_tensors in character_tensors:
            # Create a copy of the current character tensors to avoid modifying the original list
            padded_char_tensors = char_tensors.copy()
            # Append the padding tensor until the desired length is reached
            while len(padded_char_tensors) < max_line_length:
                padded_char_tensors.append(
                    torch.tensor(PADDING_TENSOR_CHARACTER)
                )  # Do i need to do another torch.tensor here?
            # Add the padded character tensors to the final list
            padded_character_tensors.append(padded_char_tensors)

        # Now create a nice large dictionary with per line its meter and the tensors we created
        print("Creating dictionary.")
        self.list_with_poetry_objects = poetry_line_per_row_df.select(
            "meter"
        ).to_dicts()
        for i, dictionary in enumerate(self.list_with_poetry_objects):
            dictionary["syllables"] = padded_syllable_tensors[i]
            dictionary["words"] = padded_word_tensors[i]
            dictionary["labels"] = padded_label_tensors[i]
            dictionary["characters"] = padded_character_tensors[i]
