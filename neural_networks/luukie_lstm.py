from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
import torch
import polars as pl
from tqdm import tqdm

from neural_networks.bilstm_crf import BiLSTM_CRF
class LuukieLSTM:
    def run(self) -> None:
        # Read the dataframe
        print('Reading parquet file.')
        df = pl.read_parquet('datalake/bucket/enriched/poetry/poetry_dataframe.parquet')
        df = df.filter(pl.col("meter") == "hexameter")
        # In the dataframe we have a row per syllable: we want a row per line of poetry, so group by line number.
        poetry_line_per_row_df = (
            df
            .group_by("line_number")
            .agg([
                pl.col("syllable").alias("syllables"),
                pl.col("label").alias("labels"),
                pl.col("word").alias("words")
            ])
        )

        # An LSTM can only accept integers, so we use one-hot encoding to turn our strings into integers.
        # We do this for all our inputs and our labels.
        print('Running encoders.')
        syllable_encoder = LabelEncoder()
        syllable_encoder.fit(df['syllable'].to_list() + ['PADDING'])

        label_encoder = LabelEncoder()
        label_encoder.fit(df['label'].to_list() + ['PADDING'])

        word_encoder = LabelEncoder()
        word_encoder.fit(df['word'].to_list() + ['PADDING'])
        
        # Now for each line of poetry, we must one-hot encode its syllables, words and labels.
        # So per line in our dataframe, use the encode to turn [ar, ma] into e.g. [12, 14].
        print('Preparing sequences.')
        syllable_sequences = []
        word_sequences = []
        label_sequences = []
        
        # Create lookups to speedup the creation of tensors.  
        word_to_id = {word: idx for idx, word in enumerate(word_encoder.classes_)}
        syllable_to_id = {syllable: idx for idx, syllable in enumerate(syllable_encoder.classes_)}
        label_to_id = {label: idx for idx, label in enumerate(label_encoder.classes_)}

        print('Number of iterations to do:', poetry_line_per_row_df.height)
        for row in tqdm(poetry_line_per_row_df.iter_rows(named=True)):
            # Create a lookup for all words and then create a tensor, where words are now encoded as integers.
            ids = [word_to_id.get(word, 'UNKNOWN') for word in row['words']]
            word_tensor = torch.tensor(ids)
            word_sequences.append(word_tensor)

            ids = [syllable_to_id.get(syllable, 'UNKNOWN') for syllable in row['syllables']]
            syllable_tensor = torch.tensor(ids)
            syllable_sequences.append(syllable_tensor)

            ids = [label_to_id.get(label, 'UNKNOWN') for label in row['labels']]
            label_tensor = torch.tensor(ids)
            label_sequences.append(label_tensor)

        # Now we have for every line of poetry a list of one-hot encoded syllables, words and labels for that list.
        # For example, for the line [ar, ma] we have three lists: [12, 14], [55, 55], [0, 1] for syllables, words and labels.
        # An LSTM wants a lot of lines of the same length, so we use padding to make every line the same length.
        PAD_SYL = syllable_encoder.transform(['PADDING'])[0]
        PAD_WORD = word_encoder.transform(['PADDING'])[0]
        PAD_LABEL = label_encoder.transform(['PADDING'])[0]

        syllable_padded = pad_sequence(syllable_sequences, batch_first=True, padding_value=PAD_SYL)
        word_padded = pad_sequence(word_sequences, batch_first=True, padding_value=PAD_WORD)
        label_padded = pad_sequence(label_sequences, batch_first=True, padding_value=PAD_LABEL)
        # If forgot what the mask does.
        mask_all = syllable_padded != PAD_SYL

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
