def do_prediction(self, 
    model, 
    X: list, 
    y: list
    ):
    """Does a prediction given the model, X and y sets

    Args:
        model (tensorflow model): that needs to do the predicting
        X (list): with syllables that need predicting
        y (list): with labels that are the predictions
    """        
    y_pred = model.predict(X)

    if self.anceps_label: # We dont want to predict the label anceps, so we delete it from the possible predictions
        for line in y_pred:
            for syllable in line:
                # check which label has the highest confidence
                position = np.where(syllable == np.amax(syllable))
                # if it is the anceps label, check confidence for long and short
                if position[0][0] == 5:
                    long = syllable[0]
                    short = syllable[1]
                    if long > short:
                        syllable = np.array([1, 0, 0, 0, 0, 0])
                    else:
                        syllable = np.array([0, 1, 0, 0, 0, 0])                        
    # generate y_pred and return per syllable the integer (syllable) with the highest confidence            
    y_pred = np.argmax(y_pred, axis=-1)
    y = np.argmax(y, axis=-1)

    return y, y_pred

def create_confusion_matrix(self, 
    model, 
    X : list, 
    y : list):
    """Creates a confusion matrix from the given model, X and y sets. As y is one-hot encoded, we need to take
    the argmax value. Secondly, because the list of lists structure (sentences), we need to flatten both prediction
    lists in order to pass them to the confusion_matrix function.

    Args:
        model (object): of lstm model
        X (list): list of lists with sentences encoded as integers
        y (list): of labels, same as X.

    Returns:
        confusion matrix: of labels
    """        
    if self.FLAGS.anceps: # We dont want to predict the label anceps, so we delete it from the possible predictions
        y_pred = y_pred[:, :, :-1]

    y, y_pred = self.do_prediction(model, X, y)

    return sklearn.metrics.confusion_matrix(self.flatten_list(y), self.flatten_list(y_pred))

def create_classification_report(self, 
    model, 
    X : list, 
    y : list, 
    output_dict : dict = True):
    """Returns a metrics classification report given the model and X and y sets.
    This shows the precision and recall of label predictions

    Args:
        model (object): the given model, LSTM in this case
        X (list): with training examples
        y (list): with training labels

    Returns:
        dataframe: with the metrics report to be printed
    """        
    y, y_pred = self.do_prediction(model, X, y)
    return sklearn.metrics.classification_report(self.flatten_list(y), self.flatten_list(y_pred), 
        # labels=[0, 1, 2, 3, 4],
        # target_names=['long', 'short', 'elision', 'space', 'padding'],
        output_dict=True, 
    )