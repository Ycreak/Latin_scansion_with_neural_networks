# Import gensim 
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import normalize

class Word_vector_creator:
    def __init__(self, character_list, vector_size, window_size):
        """This class expects a list of characters and converts the characters into a gensim model.
        The model is then saved to disk for later use.

        Args:
            list (list): of characters to be converted into a word2vec model

        Returns:
            object: gensim word2vec model
        """
        self.model = self.Generate_Word2Vec_model(character_list, vector_size, window_size)              

    def Generate_Word2Vec_model(self, vector_list, vector_size, window_size):
        """Returns Word2Vec model generated from the given text-list

        Args:
            vectorList (list): of words of the text we want to create the model on
            size (int): of dimension
            window (int): size of window

        Returns:
            object: the word2vec model
        """
        # vector_size = vector_size should be used. Older gensim versions want size=vector_size    
        return Word2Vec([vector_list], size=vector_size, window=window_size, min_count=1, workers=4)


if __name__ == "__main__":

    import utilities as util
    import random
    import numpy as np

    def find_most_frequent_label():
        """This functions finds the most common label from the given dataframe
        """          
        import pandas as pd
        import json

        threshold_elision = 0.2
        threshold_short = 0.5

        # Create our output dataframe
        column_names = ["syllable", "length"]
        new_df = pd.DataFrame(columns = column_names)
        # Load the dataframe
        df = util.Pickle_read(util.cf.get('Pickle', 'df_pedecerto_path'), 'pedecerto_df_VERG-aene.pickle')
        # Get a list of unique syllables to loop through
        unique_syllables = df['syllable'].unique()
        for syllable in unique_syllables:
            df_temp = df.loc[df['syllable'] == syllable]
            # For every syllable dataframe, create a json object with value counts
            result = json.loads(df_temp['length'].value_counts(normalize=True).to_json())
            # It now shows for every label its ratio of occurences: value between 0 and 1.
            # Labels: 0 -> short, 1 -> long, 2 -> elision
            # Now compare this ratio to the threshold: if it is higher, we label it as such
            if result.get('2') and result.get('2') >= threshold_elision:
                chosen_length = 'elision'
            elif result.get('0') and result.get('0') >= threshold_short:
                chosen_length = 'short'
            else:
                chosen_length = 'long'
            # Write the result to a new dataframe line and append
            new_line = {'syllable': syllable, 'length': chosen_length}
            new_df = new_df.append(new_line, ignore_index=True)

        print(new_df)
        # Save the dataframe for later use
        util.Pickle_write('./pickle/','syllable_most_length.pickle', new_df)

    def create_dimensionality_reduction_plot(model, method, dim):
        """This function takes a word vector model and plots the given word embeddings in 2D.
        Supports tsne and pca as methods.

        Args:
            model (object): word vector object
            method (string): either 'pca' or 'tsne': reduction method to be used

        Raises:
            ValueError: if unknown method is specified
        """    
        label_points = False

        labels = []
        tokens = []
        for word in list(model.wv.index2word):
            if random.random() <= 1:
                tokens.append(model[word])
                labels.append(word)

        if method == 'tsne':
            tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=42)
            new_values = tsne_model.fit_transform(tokens)
        elif method == 'pca':
            pca_model = PCA()
            new_values = pca_model.fit_transform(tokens)       
        else:
            raise ValueError('Unknown reduction method specified.')

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        # Get the colours for our plot.
        color_df = util.Pickle_read('./pickle/','syllable_most_length.pickle')
        color_df['length'] = np.where(color_df['length'] == 'short', 'blue', color_df['length'])
        color_df['length'] = np.where(color_df['length'] == 'long', 'red', color_df['length'])
        color_df['length'] = np.where(color_df['length'] == 'elision', 'green', color_df['length'])

        plt.figure(figsize=(16, 16))
        for i in range(len(x)):
            my_colour = color_df.loc[color_df['syllable'] == labels[i], 'length'].iloc[0]

            plt.scatter(x[i],y[i], color=my_colour)
            
            if label_points:
                plt.annotate(labels[i],
                            xy=(x[i], y[i]),
                            xytext=(5, 2),
                            textcoords='offset points',
                            ha='right',
                            va='bottom',
                            color='black')

        import matplotlib.patches as mpatches

        red_patch = mpatches.Patch(color='red', label='long')
        blue_patch = mpatches.Patch(color='blue', label='short')
        green_patch = mpatches.Patch(color='green', label='elision')

        plt.legend(handles=[red_patch, blue_patch, green_patch], loc='lower right')

        title = method + str(dim)

        plt.title(title)

        save_string = './plots/' + method + str(dim) + '.png'

        plt.savefig(save_string)

        # plt.show()

        plt.clf()

    # corpus = util.Pickle_read('/home/luukie/Github/LatinScansionModel/experimental/fasttext/','word2vec_model.pickle')
    # model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=500, workers=4)
    
    # find_most_frequent_label()
    verg_character_list = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'char_list'))
    word2vec_creator = Word_vector_creator(verg_character_list, 25, 5)
    create_dimensionality_reduction_plot(word2vec_creator.model, 'pca', 25)

    # model = util.Pickle_read('/home/luukie/Github/LatinScansionModel/experimental/fasttext/','word2vec_model.pickle')


