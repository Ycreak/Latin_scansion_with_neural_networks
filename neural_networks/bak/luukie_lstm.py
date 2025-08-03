from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
import torch
import polars as pl
from tqdm import tqdm

from neural_networks.bilstm_crf import BiLSTM_CRF
from neural_networks.common_lstm_functions import get_encoder

import torch
import torch.nn as nn

class SimpleSyllableLSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, pad_idx, emb_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, tagset_size)  # bidirectional = 2x

    def forward(self, x):
        x = self.embedding(x)                    # [B, T, emb_dim]
        lstm_out, _ = self.lstm(x)               # [B, T, 2*hidden_dim]
        logits = self.classifier(lstm_out)       # [B, T, num_labels]
        return logits


class LuukieLSTM:

    def __init__(self) -> None:
        # Padding string used in the LSTM to make all lines the same length.
        self.PADDING = 'PADDING'

    def _get_encoder(self, string_list_to_encode: list[str]) -> LabelEncoder:
        """
        Returns the encoder needed to encode strings in a list to integers.
        :param string_list_to_encode (list[str])
        :return LabelEncoder
        """
        return LabelEncoder().fit(string_list_to_encode)

    def decode_line(self) -> None:
        # Print first line of padded inputs (index 0)
        line_index = 0


        # Raw tensors
        syllable_ids = syllable_padded[line_index]
        label_ids = label_padded[line_index]
        mask = mask_all[line_index]

        # Decode only non-padding values
        decoded_syllables = [syllable_encoder.classes_[id] for id, m in zip(syllable_ids, mask) if m]
        decoded_labels = [label_encoder.classes_[id] for id, m in zip(label_ids, mask) if m]

        print("\n📝 First line of padded input and labels:")
        for i, (syl, lbl) in enumerate(zip(decoded_syllables, decoded_labels)):
            print(f"{i:2d}: {syl:15} → {lbl}")        

    def run(self) -> None:
        # Read the dataframe
        print('Reading parquet file.')
        df = pl.read_parquet('datalake/bucket/enriched/poetry/poetry_dataframe.parquet')
        df = df.filter(pl.col("meter") == "hexameter")
        # In the dataframe we have a row per syllable: we want a row per line of poetry, so group by line number.
        print('Building dataframe with one row per line of poetry.')
        poetry_line_per_row_df = (
            df
            .group_by("line_number")
            .agg([
                pl.col("syllable").alias("syllables"),
                pl.col("label").alias("labels"),
                pl.col("word").alias("words")
            ])
        )

        poetry_line_per_row_df = poetry_line_per_row_df.head(5000)

        # Now we need all our features and labels in separate lists, including the padding string.
        all_unique_syllables: list[str] = df['syllable'].to_list() + [self.PADDING]
        all_unique_labels: list[str] = df['label'].to_list() + [self.PADDING]
        all_unique_words: list[str] = df['word'].to_list() + [self.PADDING]

        # An LSTM can only accept integers, so we use one-hot encoding to turn our strings into integers.
        # We do this for all our inputs and our labels. 
        print('Getting encoders.')
        syllable_encoder: LabelEncoder = self._get_encoder(all_unique_syllables)
        label_encoder: LabelEncoder = self._get_encoder(all_unique_labels)
        word_encoder: LabelEncoder = self._get_encoder(all_unique_words)

        # Now for each line of poetry, we must one-hot encode its syllables, words and labels.
        # So per line in our dataframe, use the encode to turn [ar, ma] into e.g. [12, 14].
        print('Preparing sequences.')
        syllable_tensors: list[Tensor] = []
        word_tensors: list[Tensor] = []
        label_tensors: list[Tensor] = []

        # Create lookups to speedup the creation of tensors.  
        word_to_id = {i: idx for idx, i in enumerate(word_encoder.classes_)}
        syllable_to_id = {i: idx for idx, i in enumerate(syllable_encoder.classes_)}
        label_to_id = {i: idx for idx, i in enumerate(label_encoder.classes_)}

        print('Number of iterations to do:', poetry_line_per_row_df.height)
        for row in tqdm(poetry_line_per_row_df.iter_rows(named=True)):
            # Use the lookup for each feature and the labels to one-hot encode our strings.
            ids = [word_to_id.get(word) for word in row['words']]
            word_tensor: Tensor = torch.tensor(ids)
            word_tensors.append(word_tensor)

            ids = [syllable_to_id.get(syllable) for syllable in row['syllables']]
            syllable_tensor: Tensor = torch.tensor(ids)
            syllable_tensors.append(syllable_tensor)

            ids = [label_to_id.get(label) for label in row['labels']]
            label_tensor: Tensor = torch.tensor(ids)
            label_tensors.append(label_tensor)

        # Now we have for every line of poetry a list of one-hot encoded syllables, words and labels for that list.
        # For example, for the line [ar, ma] we have three lists: [12, 14], [55, 55], [0, 1] for syllables, words and labels.
        # An LSTM wants a lot of lines of the same length, so we use padding to make every line the same length.
        PADDING_INTEGER_SYLLABLE = syllable_encoder.transform(['PADDING'])[0]
        PADDING_INTEGER_WORD = word_encoder.transform(['PADDING'])[0]
        PADDING_INTEGER_LABEL = label_encoder.transform(['PADDING'])[0]

        padded_syllable_tensors = pad_sequence(syllable_tensors, batch_first=True, padding_value=PADDING_INTEGER_SYLLABLE)
        padded_word_tensors = pad_sequence(word_tensors, batch_first=True, padding_value=PADDING_INTEGER_WORD)
        padded_label_tensors = pad_sequence(label_tensors, batch_first=True, padding_value=PADDING_INTEGER_LABEL)

        # If forgot what the mask does.
        mask_all = syllable_padded != PAD_SYL

        model = SimpleSyllableLSTM(
            vocab_size=len(syllable_encoder.classes_),
            tagset_size=len(label_encoder.classes_),
            pad_idx=PAD_SYL
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)

        for epoch in range(50):
            model.train()
            logits = model(syllable_padded)                         # [B, T, C]
            loss = criterion(logits.view(-1, logits.size(-1)),      # flatten [B*T, C]
                            label_padded.view(-1))                 # flatten [B*T]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            logits = model(syllable_padded)                  # [B, T, C]
            predictions = torch.argmax(logits, dim=-1)       # [B, T]

        decoded_preds = [
            [label_encoder.classes_[label.item()] for label, m in zip(line, mask_line) if m]
            for line, mask_line in zip(predictions, syllable_padded != PAD_SYL)
        ]

        # Flatten predictions and labels, ignoring padding
        true_labels = []
        pred_labels = []

        for pred_seq, true_seq, mask_seq in zip(predictions, label_padded, syllable_padded != PAD_SYL):
            for pred, true, mask in zip(pred_seq, true_seq, mask_seq):
                if mask:
                    true_labels.append(label_encoder.classes_[true.item()])
                    pred_labels.append(label_encoder.classes_[pred.item()])

        # Print classification report
        print(classification_report(true_labels, pred_labels, digits=4))

        # exit(0)

        #####################################
         
        # Create the model
        # embedding_output_dim: int = 25 # 25      
        # lstm_layer_units: int = 10 # 10
        # LEARNING_RATE: float = 0.001
            
        # # Initiate the model structure
        # input_layer = tf.keras.layers.Input(shape=(len(syllable_padded[0]),))
        
        # model = tf.keras.layers.Embedding(
        #     input_dim = len(syllable_encoder.classes_), 
        #     output_dim = embedding_output_dim, 
        #     input_length = len(syllable_padded[0])
        # )(input_layer)

        # # model = Dropout(0.1)(model)

        # # model = tf.keras.layers.LSTM(
        # #         units = lstm_layer_units, 
        # #         return_sequences = True, 
        # #         recurrent_dropout = 0.1
        # #     )(model)

        # model = tf.keras.layers.Bidirectional(
        #     tf.keras.layers.LSTM(
        #         units = lstm_layer_units, 
        #         return_sequences = True, 
        #         recurrent_dropout = 0.1
        #     )
        # )(model)

        # model = tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')(model)

        # # output_layer = TimeDistributed(Dense(50, activation="softmax"))(model)  # softmax output layer
        # # kernel = TimeDistributed(Dense(num_labels, activation="softmax"))(model)  # softmax output layer

        # model = tf.keras.models.Model(input_layer, model)

        # model.compile(
        #     # optimizer="rmsprop",
        #     optimizer=tf.keras.optimizers.Adam(
        #         learning_rate = LEARNING_RATE
        #     ),  # Optimizer
        #     # Loss function to minimize
        #     loss=tf.keras.losses.CategoricalCrossentropy(),
        #     # loss=tfa.losses.SigmoidFocalCrossEntropy(),
        #     # List of metrics to monitor
        #     # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        #     # metrics=tf.keras.metrics.CategoricalAccuracy(),
        #     metrics=['accuracy'],
        # )

        # print(model.summary())

        # history = model.fit(
        #     syllable_padded, 
        #     label_padded, 
        #     batch_size = 32, 
        #     epochs = 25, 
        #     validation_split = 0.2, 
        #     verbose = True
        # )
            
        # exit(0)
        ##################################

        # K-Fold Cross-Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(kf.split(syllable_padded)):
            print(f"\n=== Fold {fold + 1} ===")

            syllable_train = syllable_padded[train_idx]
            syllable_test = syllable_padded[test_idx]

            word_train = word_padded[train_idx]
            word_test = word_padded[test_idx]

            label_train = label_padded[train_idx]
            label_test = label_padded[test_idx]

            mask_train = mask_all[train_idx]
            mask_test = mask_all[test_idx]

            model = BiLSTM_CRF(
                syll_vocab_size=len(syllable_encoder.classes_),
                word_vocab_size=len(word_encoder.classes_),
                tagset_size=len(label_encoder.classes_),
                PAD_SYL=PAD_SYL, PAD_WORD=PAD_WORD
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(25):  # Adjust as needed
                model.train()
                loss = model(syllable_train, word_train, tags=label_train, mask=mask_train)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f"Fold {fold + 1} Epoch {epoch} Loss: {loss.item():.4f}")

            model.eval()
            with torch.no_grad():
                predictions = model(syllable_test, word_test, mask=mask_test)

            # Decode predictions and syllables for the first test sample
            sample_idx = 0  # change if you want to inspect a different line
            sample_pred = predictions[sample_idx]
            sample_syllables = syllable_test[sample_idx]
            sample_mask = mask_test[sample_idx]

            # Decode to strings using encoders
            decoded_preds = [label_encoder.classes_[idx] for idx, m in zip(sample_pred, sample_mask) if m]
            decoded_sylls = [syllable_encoder.classes_[idx] for idx, m in zip(sample_syllables, sample_mask) if m]

            # Print side-by-side
            print(f"\nFold {fold + 1} Sample Line:")
            for syll, label in zip(decoded_sylls, decoded_preds):
                print(f"  {syll:10} -> {label}")

            # And print a confusion matrix
            all_preds = []
            all_trues = []

            for pred_line, true_line, mask_line in zip(predictions, label_test, mask_test):
                for pred_token, true_token, m in zip(pred_line, true_line, mask_line):
                    if m:  # skip padding
                        all_preds.append(pred_token)
                        all_trues.append(true_token.item())

            # Convert to label strings
            decoded_preds = [label_encoder.classes_[idx] for idx in all_preds]
            decoded_trues = [label_encoder.classes_[idx] for idx in all_trues]

            # Print classification report
            print(f"\n=== Evaluation for Fold {fold + 1} ===")
            print(classification_report(decoded_trues, decoded_preds, digits=4))
            print(f"Accuracy: {accuracy_score(decoded_trues, decoded_preds):.4f}")
            exit(0)

if __name__ == "__main__":
    lstm = LuukieLSTM()
    lstm.run()
