from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import FlairEmbeddings
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.datasets import UD_ENGLISH
from flair.embeddings import WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, FastTextEmbeddings
# from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

import numpy as np
from sklearn.model_selection import train_test_split

import utilities as util
import argparse
import functools

class FLAIR_model():

    def __init__(self, FLAGS):

        self.corpus_path = './flair/corpus/trimeter' #'./flair/corpus'
        self.flair_lm_path = './flair/resources/taggers/dactylic_model'
        self.flair_lm_output = 'flair/resources/taggers/iambic_model' #'flair/resources/taggers/dactylic_model'

        # FLAGS.custom_prediction = True

        if FLAGS.create_corpus:
            trimeter = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), 'SEN-precise.pickle')
            self.create_corpus_files(trimeter, self.corpus_path)

            # Creates corpus files from the given sequence label lists to allow FLAIR to do its training on
            # all_sequence_label_pickles = util.Create_files_list(util.cf.get('Pickle', 'path_sequence_labels'), 'pickle') # Find all pickle files
            # sequence_labels_all_set = util.merge_sequence_label_lists(all_sequence_label_pickles, util.cf.get('Pickle', 'path_sequence_labels')) # Merge them into one big file list
            # self.create_corpus_files(sequence_labels_all_set, self.corpus_path)
            
        if FLAGS.create_syllable_file:
            # Creates a plain text file of syllables to train word embeddings on later
            all_sequence_label_pickles = util.Create_files_list(util.cf.get('Pickle', 'path_sequence_labels'), 'pickle') # Find all pickle files
            sequence_labels_all_set = util.merge_sequence_label_lists(all_sequence_label_pickles, util.cf.get('Pickle', 'path_sequence_labels')) # Merge them into one big file list           
            self.create_plain_syllable_files(sequence_labels_all_set, './flair/corpus_in_plain_syllables.txt')

        if FLAGS.train_model:
            # trains and saves the FLAIR model
            self.train_model(self.corpus_path, self.flair_lm_path, self.flair_lm_output)

        if FLAGS.language_model == 'flair':
            # Creates the flair language model by training embeddings on the text
            self.train_flair_language_model(self.corpus_path, self.flair_lm_path)

        if FLAGS.language_model == 'fasttext':
            import fasttext
            # Creates the fasttext language model by training embeddings on the given text
            model = fasttext.train_unsupervised(input = 'flair/corpus_in_plain_syllables.txt',
                                                model = 'skipgram',
                                                lr = 0.05,
                                                dim = 100,
                                                ws = 5,                 # window size
                                                epoch = 5,
                                                minCount = 1,
                                                verbose = 2,
                                                thread = 4
                                                )

            model.save_model("flair/resources/fasttext_embeddings.bin")            

        if FLAGS.custom_prediction:

            custom_prediction_corpus_path = './flair/corpus/custom_prediction'
            custom_prediction_model_output_path = './flair/resources/taggers/custom_prediction_model'

            language_model_path = './flair/resources/taggers/custom_flair_language_model'

            # corpus_text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), 'SEN-precise.pickle')

            # self.create_corpus_files(corpus_text, custom_prediction_corpus_path)

            self.train_flair_language_model(custom_prediction_corpus_path, language_model_path)

            self.train_model(custom_prediction_corpus_path, language_model_path, custom_prediction_model_output_path)

            _predictor = custom_prediction_model_output_path + '/final-model.pt'

            self.custom_prediction( predictor=_predictor,
                                    predictee=util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'),'SEN-aga.pickle'),
                                    )
                
                
        if FLAGS.single_line:
            # create example sentence
            sentence = Sentence('de lu bra - et - a ras - cae li tum - et - pa tri os - la res')
            print(sentence.labels)
            print(sentence.to_tagged_string())         

        if FLAGS.qualitative:
            # Qualitative research
            custom_prediction_model_output_path = './flair/resources/taggers/custom_prediction_model'
            predictor = custom_prediction_model_output_path + '/final-model.pt'
            predictee = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'),'SEN-aga.pickle')
            # load the model you trained
            model = SequenceTagger.load(predictor)

            counter = 0

            # Loop through each line and get the ground truth and sentence from the given text
            for line in predictee:
                y_true = []
                y_pred = []
                new_line = ''
                for syllable, label in line:
                    new_line += syllable + ' '
                    y_true.append(label)
                # Turn the string into a FLAIR sentence and let the model predict
                sentence = Sentence(new_line.rstrip())
                model.predict(sentence, all_tag_prob=True)
                # Create a y_pred list
                for token in sentence.tokens:
                    
                    # i take a look at the probability distribution for the labels.
                    proba = token.tags_proba_dist['length']
                    # i put this in a list i can work with (no idea what flair is doing)
                    probability_list = []
                    for prob in proba:
                        probability_list.append((prob.value, prob.score))
                    # next find the highest probability tuple
                    highest_tuple = max(probability_list, key = lambda i : i[1])
                    # if it is an anceps, we do not want to predict it. we want to predict short or long                
                    if highest_tuple[0] == 'anceps':
                        # pick highest confidence short or long
                        long_label_confidence = [t[1] for t in probability_list if t[0] == 'long']
                        short_label_confidence = [t[1] for t in probability_list if t[0] == 'short']
                        if long_label_confidence > short_label_confidence:
                            y_pred.append('long')
                        else:
                            y_pred.append('short')
                    # if no anceps, just continue with the highest confidence tuple
                    else:
                        y_pred.append(highest_tuple[0]) # append the label with highest confidence            

                # print(type(y_pred))

                # if not (y_pred_current == y_true_current).all():
                if not self.check_lists_indentically(y_pred, y_true):
                    counter += 1
                    print('### THIS LINE CONTAINS ERRORS ###')
                    print(sentence.tokenized)         
                    print('TRUE', y_true)
                    print('PRED', y_pred)
                    print('#################################\n')
                else:
                    print('@@@ THIS LINE IS CORRECT @@@')
                    print(sentence.tokenized)         
                    print('TRUE', y_true)
                    print('PRED', y_pred)
                    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')

            print(counter)

    def check_lists_indentically(self, list1: list, list2: list):
        # Returns whether two given lists are identical (item for item)
        import functools
        if functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, list1, list2), True):
            return True
        else:
            return False 



    def remove_anceps_from_syllable_sequence(self, given_list):
        return [i for i in given_list if i[0] != 'anceps']

    def custom_prediction(self, predictor, predictee):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        import pandas as pd

        # load the model you trained
        model = SequenceTagger.load(predictor)

        y_true = []
        y_pred = []

        # Loop through each line and get the ground truth and sentence from the given text
        for line in predictee:
            new_line = ''
            for syllable, label in line:
                new_line += syllable + ' '
                y_true.append(label)
            # Turn the string into a FLAIR sentence and let the model predict
            sentence = Sentence(new_line.rstrip())
            model.predict(sentence, all_tag_prob=True)
            # Create a y_pred list
            for token in sentence.tokens:
                
                # i take a look at the probability distribution for the labels.
                proba = token.tags_proba_dist['length']
                # i put this in a list i can work with (no idea what flair is doing)
                probability_list = []
                for prob in proba:
                    probability_list.append((prob.value, prob.score))
                # next find the highest probability tuple
                highest_tuple = max(probability_list, key = lambda i : i[1])
                # if it is an anceps, we do not want to predict it. we want to predict short or long                
                if highest_tuple[0] == 'anceps':
                    # pick highest confidence short or long
                    long_label_confidence = [t[1] for t in probability_list if t[0] == 'long']
                    short_label_confidence = [t[1] for t in probability_list if t[0] == 'short']
                    if long_label_confidence > short_label_confidence:
                        y_pred.append('long')
                    else:
                        y_pred.append('short')
                # if no anceps, just continue with the highest confidence tuple
                else:
                    y_pred.append(highest_tuple[0]) # append the label with highest confidence

        # From all y_true and y_pred, create a confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['long', 'short', 'elision', 'space'])

        # Additionally create a metrics report
        metrics_report = classification_report(y_true, y_pred, digits=4)
        print(metrics_report)

        df_confusion_matrix = pd.DataFrame(cm, index = ['long', 'short', 'elision', 'space'],
                                            columns = ['long', 'short', 'elision', 'space'])
        # Drop the padding labels, as we don't need them (lstm scans them without confusion): delete both row and column
        df_confusion_matrix = df_confusion_matrix.drop(columns=['space'])
        df_confusion_matrix = df_confusion_matrix.drop('space')

        # Create a heatmap and save it to disk
        util.create_heatmap(dataframe = df_confusion_matrix,
                            xlabel = 'TRUTH', 
                            ylabel =  'PREDICTED',
                            title = 'CONFUSION MATRIX FLAIR',
                            filename = 'confusion_matrix_flair_seneca',
                            ) 

    def create_plain_syllable_files(self, sequence_labels, filename):
        """Writes all syllables from the given sequence label file to a text file
        with the given name. Stores these in root. 

        Args:
            sequence_labels (list): with sequence labels
            filename (string): name of the save file
        """
        file = open(filename, 'w')
        for sentence in sequence_labels:
            for syllable, label in sentence:
                result = syllable + ' '
                file.write(result)
            file.write('\n')
        file.close()    

    def create_corpus_files(self, sequence_labels, corpus_path, test_split=0.2, validate_split=0.1):
        """Creates Flair corpus files given the sequence labels. Saves the train, test and validation
        files to the corpus folder in root.

        Args:
            sequence_labels (list): with sequence labels
        """
        train_path = corpus_path + '/train.txt'
        test_path = corpus_path + '/test.txt'
        valid_path = corpus_path + '/valid.txt'

        train_file = open(train_path, 'w') # NEEDS TO BE ITS OWN FOLDER...
        test_file = open(test_path, 'w')
        validate_file = open(valid_path, 'w')
        # Create the train, test and validate splits
        X_train, X_test = train_test_split(sequence_labels, test_size=test_split, random_state=42)
        X_train, X_validate = train_test_split(X_train, test_size=validate_split, random_state=42)

        for sentence in X_train:
            for syllable, label in sentence:
                result = syllable + '\t' + label + '\n'
                train_file.write(result)
            train_file.write('\n')
        train_file.close()

        for sentence in X_test:
            for syllable, label in sentence:
                result = syllable + '\t' + label + '\n'
                test_file.write(result)
            test_file.write('\n')
        test_file.close()

        for sentence in X_validate:
            for syllable, label in sentence:
                result = syllable + '\t' + label + '\n'
                validate_file.write(result)
            validate_file.write('\n')
        validate_file.close()

    def train_flair_language_model(self, corpus_path, output_path):
        """This function trains a Flair language model and saves it to disk for later use
        """    

        #TODO: this function wants train.txt in train/train.txt for some reason.

        # are you training a forward or backward LM?
        is_forward_lm = True
        # load the default character dictionary
        dictionary: Dictionary = Dictionary.load('chars')
        # get your corpus, process forward and at the character level
        corpus = TextCorpus(corpus_path,
                            dictionary,
                            is_forward_lm,
                            character_level=True)

        # instantiate your language model, set hidden size and number of layers
        language_model = LanguageModel(dictionary,
                                    is_forward_lm,
                                    hidden_size=128,
                                    nlayers=1)
        # train your language model
        trainer = LanguageModelTrainer(language_model, corpus)

        trainer.train(output_path,
                    sequence_length=10,
                    mini_batch_size=10,
                    max_epochs=FLAGS.epochs)

    def load_corpus(self, corpus_path):
        """This function loads a corpus from disk and returns it

        Returns:
            Corpus: with train, test and validation data
        """    
        columns = {0: 'syllable', 1: 'length'}
        # data_folder = 'flair/corpus'
        # init a corpus using column format, data folder and the names of the train, dev and test files
        corpus: Corpus = ColumnCorpus(corpus_path, columns,     # omg a type hint in python
                                    train_file='train.txt',
                                    test_file='test.txt',
                                    dev_file='valid.txt')
        return corpus

    def train_model(self, corpus_path, flair_path, model_output_path):
        # 1. get the corpus
        corpus = self.load_corpus(corpus_path)
        # 2. what label do we want to predict?
        label_type = 'length'
        # 3. make the label dictionary from the corpus
        label_dictionary = corpus.make_label_dictionary(label_type='length')
        
        # # 4. initialize embeddings
        chosen_flair_model = flair_path + '/best-lm.pt'

        embedding_types = [
            # GLOVE EMBEDDINGS
            # WordEmbeddings('glove'),
            
            # FASTTEXT EMBEDDINGS
            # FastTextEmbeddings('flair/resources/fasttext_embeddings.bin'),
            
            # GENSIM EMBEDDINGS
            # custom_embedding = WordEmbeddings('path/to/your/custom/embeddings.gensim')
            
            # CHARACTER EMBEDDIGS
            CharacterEmbeddings(),

            # FLAIR EMBEDDINGS
            FlairEmbeddings(chosen_flair_model),
        ]
        embeddings = StackedEmbeddings(embeddings=embedding_types)

        # 5. initialize sequence tagger
        tagger = SequenceTagger(hidden_size=256,
                                embeddings=embeddings,
                                tag_dictionary=label_dictionary,
                                tag_type=label_type,
                                use_crf=True)
        # 6. initialize trainer 
        trainer = ModelTrainer(tagger, corpus)
        # 7. start training and save to disk
        trainer.train(model_output_path,
                    learning_rate=0.1,
                    mini_batch_size=32,
                    max_epochs=FLAGS.epochs)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--create_model", action="store_true", help="specify whether to create the model: if not specified, we load from disk")
    p.add_argument("--save_model", action="store_true", help="specify whether to save the model: if not specified, we do not save")
    p.add_argument("--train_model", action="store_true", help="specify whether to train a FLAIR model")
    p.add_argument("--create_corpus", action="store_true", help="specify whether to create the corpus for FLAIR")
    p.add_argument("--create_syllable_file", action="store_true", help="specify whether to create a file consisting of syllables to train word vectors on")
    p.add_argument("--custom_prediction", action="store_true", help="specify whether to test the FLAIR model")
    p.add_argument("--exp_train_test", action="store_true", help="specify whether to run the train/test split LSTM experiment")
    p.add_argument("--single_line", action="store_true", help="specify whether to predict a single line")
    p.add_argument("--qualitative", action="store_true", help="specify whether to predict a single line")

    p.add_argument("--verbose", action="store_true", help="specify whether to run the code in verbose mode")
    p.add_argument('--epochs', default=10, type=int, help='number of epochs')
    p.add_argument("--split", type=util.restricted_float, default=0.2, help="specify the split size of train/test sets")

    p.add_argument('--language_model', default='none', type=str, help='name of LM to train')


    FLAGS = p.parse_args()    
    
    my_flair = FLAIR_model(FLAGS)
