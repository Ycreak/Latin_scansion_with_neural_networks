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

import fasttext

class FLAIR_model():

    def __init__(self, FLAGS):

        if FLAGS.create_corpus:
            # Creates corpus files from the given sequence label lists to allow FLAIR to do its training on
            all_sequence_label_pickles = util.Create_files_list(util.cf.get('Pickle', 'path_sequence_labels'), 'pickle') # Find all pickle files
            sequence_labels_all_set = util.merge_sequence_label_lists(all_sequence_label_pickles, util.cf.get('Pickle', 'path_sequence_labels')) # Merge them into one big file list
            self.create_corpus_files(sequence_labels_all_set)
            
        if FLAGS.create_syllable_file:
            # Creates a plain text file of syllables to train word embeddings on later
            all_sequence_label_pickles = util.Create_files_list(util.cf.get('Pickle', 'path_sequence_labels'), 'pickle') # Find all pickle files
            sequence_labels_all_set = util.merge_sequence_label_lists(all_sequence_label_pickles, util.cf.get('Pickle', 'path_sequence_labels')) # Merge them into one big file list           
            self.create_plain_syllable_files(sequence_labels_all_set, './flair/corpus_in_plain_syllables.txt')

        if FLAGS.train_model:
            # trains and saves the FLAIR model
            self.train_model()

        if FLAGS.language_model == 'flair':
            # Creates the flair language model by training embeddings on the text
            self.train_flair_language_model()

        if FLAGS.language_model == 'fasttext':
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

        if FLAGS.test_model:
            # load the model you trained
            model = SequenceTagger.load('resources/taggers/dactylic_model/final-model.pt')

            # create example sentence
            sentence = Sentence('ar ma vi rum que ca no troi ae qui pri mus ab or is')

            # predict tags and print
            model.predict(sentence)

            print(sentence.to_tagged_string())

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

    def create_corpus_files(self, sequence_labels, test_split=0.2, validate_split=0.1):
        """Creates Flair corpus files given the sequence labels. Saves the train, test and validation
        files to the corpus folder in root.

        Args:
            sequence_labels (list): with sequence labels
        """    
        train_file = open('./flair/corpus/train/train.txt', 'w') # NEEDS TO BE ITS OWN FOLDER...
        test_file = open('./flair/corpus/test.txt', 'w')
        validate_file = open('./flair/corpus/valid.txt', 'w')
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

    def train_flair_language_model(self):
        """This function trains a Flair language model and saves it to disk for later use
        """    
        # are you training a forward or backward LM?
        is_forward_lm = True
        # load the default character dictionary
        dictionary: Dictionary = Dictionary.load('chars')
        # get your corpus, process forward and at the character level
        corpus = TextCorpus('./flair/corpus',
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

        trainer.train('flair/resources/taggers/language_model',
                    sequence_length=10,
                    mini_batch_size=10,
                    max_epochs=FLAGS.epochs)

    def load_corpus(self):
        """This function loads a corpus from disk and returns it

        Returns:
            Corpus: with train, test and validation data
        """    
        columns = {0: 'syllable', 1: 'length'}
        data_folder = 'flair/corpus'
        # init a corpus using column format, data folder and the names of the train, dev and test files
        corpus: Corpus = ColumnCorpus(data_folder, columns,     # omg a type hint in python
                                    train_file='train.txt',
                                    test_file='test.txt',
                                    dev_file='valid.txt')
        return corpus

    def train_model(self):
        # 1. get the corpus
        corpus = self.load_corpus()
        # 2. what label do we want to predict?
        label_type = 'length'
        # 3. make the label dictionary from the corpus
        label_dictionary = corpus.make_label_dictionary(label_type='length')
        # 4. initialize embeddings
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
            FlairEmbeddings('flair/resources/taggers/language_model/best-lm.pt'),
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
        trainer.train('flair/resources/taggers/dactylic_model',
                    learning_rate=0.1,
                    mini_batch_size=32,
                    max_epochs=10)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--create_model", action="store_true", help="specify whether to create the model: if not specified, we load from disk")
    p.add_argument("--save_model", action="store_true", help="specify whether to save the model: if not specified, we do not save")
    p.add_argument("--train_model", action="store_true", help="specify whether to train a FLAIR model")
    p.add_argument("--create_corpus", action="store_true", help="specify whether to create the corpus for FLAIR")
    p.add_argument("--create_syllable_file", action="store_true", help="specify whether to create a file consisting of syllables to train word vectors on")
    p.add_argument("--test_model", action="store_true", help="specify whether to test the FLAIR model")
    p.add_argument("--exp_train_test", action="store_true", help="specify whether to run the train/test split LSTM experiment")
    p.add_argument("--exp_transfer_boeth", action="store_true", help="specify whether to run the Boeth LSTM experiment")

    p.add_argument("--verbose", action="store_true", help="specify whether to run the code in verbose mode")
    p.add_argument('--epochs', default=10, type=int, help='number of epochs')
    p.add_argument("--split", type=util.restricted_float, default=0.2, help="specify the split size of train/test sets")

    p.add_argument('--language_model', default='none', type=str, help='name of LM to train')


    FLAGS = p.parse_args()    
    
    my_flair = FLAIR_model(FLAGS)