import configparser
import pandas as pd
import numpy as np
from progress.bar import Bar

from tensorflow.keras import models
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

from matplotlib import pyplot

import utilities as util

class Neural_network_handler:
    """This class handles everything neural network related.
    """

    def __init__(self, df):
        # Read the config file for later use
        self.cf = configparser.ConfigParser()
        self.cf.read("config.ini")

        # Control flow booleans
        add_padding = False #True
        flatten_vector = False
        create_model = False
        test_model = True

        load_X_y = False

        # This functions add padding to every line
        if add_padding:
            print('Adding padding')
            df = self.Add_padding(df)
            util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'padded_set'), df)

        df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'padded_set'))
        if self.cf.get('Util', 'verbose'): print(df)

        if flatten_vector:
            # The network wants a single vector as input, so we flatten it for every line in the text
            print('Flattening the vectors')
            df = self.Flatten_dataframe_column(df, 'vector')
            util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'flattened_vectors'), df)
        df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'flattened_vectors'))
        if self.cf.get('Util', 'verbose'): print(df)

        ####
        # TODO: Philippe plz continue here
        ####)

        # Turn df into X and y for neural network
        print('Loading X and y')
        X, y = self.Create_X_y(df, load_X_y)
        # print("Training data: shape={}".format(X.shape))
        # print("Training target data: shape={}".format(y.shape))

        # Encode: 2 for elision and 3 for padding (0 for short, 1 for long)
        # y[y == 2] = 1 # Dont forget, y is numpy.ndarray
        # y[y == 3] = 1

        if create_model:
            ''' 
            README
            The main problem at the moment is as follows. I give a single input (ndarray), which is a line of 20 syllables, represented 
            by vectors of dimension 25 (X.shape = 500). The output i want is 20 dimensional, one for each syllable. The output i want is
            multiclass: each syllable can be short, long, elided or simply padding (encoded as 0, 1, 2 or 3). The problem is therefore 
            multiclass: each of the 20 outputs can have one and only one label. If i use binary classification and encode elision and padding
            as 1 as well, it seems to work quite well (see docs/first_four_lines.txt). If i use categorical classification and try to predict
            classes, it doesnt work as intended. Questions: can i just do multiclass prediction, or do i need to binarize my labels? Do we
            need scaling. Can we do multi output multi class the way i implemented it?
            '''

            labels = ['short', 'long', 'elision', 'padding']

            # TODO: do we need to scale the X data? All values are between -1 and 1.
            # scaler = MinMaxScaler()
            # X_train = scaler.fit_transform(X_train)
            # X_test = scaler.transform(X_test)

            #TODO: i have four labels i want to predict. do we need to binarize this?
            # mlb = MultiLabelBinarizer()
            # y = mlb.fit_transform(y)

            # one hot encode output variable (for class prediction)
            # y = to_categorical(y, num_classes=4)

            # Create and evaluate the model. Uses k-fold cross validation
            model = self.Evaluate_model(X, y)

            model.save('pickle/model')

        if test_model:
            # Load if needed. Now I just create the model every time (10 epochs)
            model = models.load_model('pickle/model')

            # TODO: i can predict using binary, but i need to predict classes. predict_classes is deprecated
            # However, the model.predict(X).argmax(axis=-1) results in the network predicting a single int64?
            # yhat = model.predict_classes(x_new)
            # yhat = model.predict(X).argmax(axis=-1) #model.predict_classes(X)
            # print([labels[i] for i in model.predict(X).argmax(axis=-1)])


            # This works fine for binary classification
            yhat = model.predict(X)

            ### Uncomment this if you want to create a prediction dataframe
            # self.Create_prediction_df(df, X, y, yhat)

            # Predict and test the first 10 lines. Also, print the similarity of predicted and expected
            for i in range(10):

                print('Expected : {0}'.format(y[i]))

                try:
                    # Round the number to the next whole number (for readability)
                    round_to_whole = [round(num) for num in yhat[i]]
                    print('Predicted: {0}'.format(round_to_whole))
                    res = self.Calculate_list_similarity(y[i], round_to_whole)
                    print('Similarity score for line {0}: {1}'.format(i, res))
                except:
                    print('Predicted: %s' % yhat[i])

                print('\n')




    def Get_model(self, n_inputs, n_outputs):
        # Creates the model and its layers. 
        _opt = 'adam'
        # TODO: this should be categorical_crossentropy, but i cant get it to work
        _loss = 'binary_crossentropy' # 'categorical_crossentropy'# _loss = 'sparse_categorical_crossentropy'

        model = Sequential()
        # Input Layer
        model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        # Hidden Layers
        model.add(Dense(16, activation='relu'))
        # Output Layer. TODO: this should be a softmax if we have categorical crossentropy
        model.add(Dense(n_outputs, activation='sigmoid')) #softmax
        # TODO: i commented the metrics, because these dont make sense at the moment 
        model.compile(loss=_loss, optimizer=_opt)# , metrics=['accuracy'])

        return model

    def Evaluate_model(self, X, y):
        # Evaluates the model using KFolds
        from sklearn.model_selection import RepeatedKFold
        from sklearn.metrics import accuracy_score

        # Neural Network parameters
        _epochs = int(self.cf.getint('NeuralNetwork', 'epochs'))
        _batch_size = int(self.cf.getint('NeuralNetwork', 'batch_size'))

        results = list()
        n_inputs, n_outputs = X.shape[1], y.shape[1]
        # define evaluation procedure
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
        # enumerate folds
        for train_ix, test_ix in cv.split(X):
            # prepare data
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            # define model
            model = self.Get_model(n_inputs, n_outputs)
            # fit model
            model.fit(X_train, y_train, verbose=1, epochs=_epochs)
            # TODO: is this below a better line? that is from the bio modeling code :)
            # history = model.fit(X_train, y_train, verbose=1, epochs=_epochs, batch_size=_batch_size,
                                # validation_data=(X_test, y_test), shuffle=True)

            # TODO: this could be nice to print, but it doesnt work at the moment
            # make a prediction on the test set
            # yhat = model.predict(X_test)
            # round probabilities to class labels
            # yhat = yhat.round()
            # calculate accuracy
            # acc = accuracy_score(y_test, yhat)
            # store result
            # print('>{0}'.format(acc))
            # results.append(acc)

            # TODO: printing accuracy, which method would be best?
            # _, train_accuracy = model.evaluate(X_train, y_train)
            # _, test_accuracy = model.evaluate(X_test, y_test)

            # print('Accuracy (training): {0}'.format(train_accuracy * 100))
            # print('Accuracy (testing): {0}'.format(test_accuracy * 100))

            # if you need debugging, you might want to use this line
            # self.Create_plots(history)

            # TODO: i return the model now after a single training round, just to get
            # everything working. Evaluate_model should just be used for tweaking settings
            return model

        return results

    def Flatten_dataframe_column(self, df, col):
        df[col] = df.apply(lambda x : np.array(x[col]).flatten(), axis=1)
        return df

    def Create_X_y(self, df, use_file):

        if use_file:
            with open('./pickle/X.npy', 'rb') as f:
                X = np.load(f, allow_pickle=True)
            with open('./pickle/y.npy', 'rb') as f:
                y = np.load(f, allow_pickle=True)

            return X, y

        X = list()
        y = list()
        for _, row in df.iterrows():
            X.append(row['vector'])
            y.append(row['length'])

        # Here I learned that tensorflow errors are unreadable. Lest we forget.
        # ValueError: setting an array element with a sequence.

        X = np.array(X, dtype=np.float)
        y = np.array(y, dtype=np.float)

        # Save our files for easy loading next time
        with open('./pickle/X.npy', 'wb') as f:
            np.save(f, X)
        with open('./pickle/y.npy', 'wb') as f:
            np.save(f, y)

        return X, y

    def Add_padding(self, df):
        """Adds padding vectors to each sentence to give the neural network a constant length per sentence.
        For example, if a sentence has 12 syllables, 8 padding vectors will be added.

        Args:
            df (dataframe): contains sentences to be padded

        Returns:
            df: with padded sentences
        """
        column_names = ["book", "line", "syllable", "length", "vector"]
        new_df = pd.DataFrame(columns = column_names)

        # same_line = True
        zero_vector = np.zeros(self.cf.getint('Word2Vec', 'vector_size'))
        max_length_sentence = self.cf.getint('NeuralNetwork', 'max_length')

        # Get number of books to process
        num_books = df['book'].max()

        for i in Bar('Processing').iter(range(num_books)):
        # for i in range(num_books): # For debugging    
            # Get only lines from this book
            current_book = i + 1
            book_df = df.loc[df['book'] == current_book]

            num_lines = book_df['line'].max()

            for j in range(num_lines):
                current_line = j + 1

                filtered_df = book_df[book_df["line"] == current_line]

                vector_list = filtered_df['vector'].tolist()
                length_list = filtered_df['length'].tolist()
                syllable_list = filtered_df['syllable'].tolist()

                # Unpack wrapped vectors from the list
                vector_list = [i.v for i in vector_list]

                # Check length of this list and add padding accordingly  
                padding_needed = max_length_sentence - len(vector_list)

                for k in range(padding_needed):
                    vector_list.append(zero_vector)
                    length_list.append(3) # Denote padding with int 3
                    syllable_list.append(0)

                # Now we can create a new pandas dataframe with all information
                new_line = {'book': current_book, 'line': current_line, 'syllable': syllable_list, 'length': length_list, 'vector': vector_list}
                new_df = new_df.append(new_line, ignore_index=True)

        return new_df

    def Calculate_list_similarity(self, list1, list2):
        # Calculates the similarity between two lists (entry for entry)
        score = self.cf.getint('NeuralNetwork', 'max_length')

        for i in range(len(list1)):

            if list1[i] != list2[i]:
                score -= 1

        return score / len(list1) * 100

    def Create_plots(self, history):
        # Does what it says on the tin

        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        # plot accuracy during training
        pyplot.subplot(212)
        pyplot.title('Accuracy')
        pyplot.plot(history.history['accuracy'], label='train')
        pyplot.plot(history.history['val_accuracy'], label='test')
        pyplot.legend()
        # pyplot.show()
        pyplot.savefig('plots/plot.png')


    def Create_prediction_df(self, df, X, y, yhat):
        # Creates a dataframe with predictions. Used by OSCC (for now)
        column_names = ["predicted", "expected"]
        new_df = pd.DataFrame(columns = column_names)

        for i in Bar('Processing').iter(range(len(X))):
            new_line = {'expected': y[i], 'predicted': yhat[i]}
            new_df = new_df.append(new_line, ignore_index=True)

        book_line_df = df[['book','line', 'syllable']]

        prediction_df = pd.concat([book_line_df, new_df], axis=1, join='inner')

        print(prediction_df)

        util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'prediction_df'), prediction_df)