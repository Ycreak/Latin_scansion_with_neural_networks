# Latin Scansion Model 2021
# This simple FLASK server interfaces with
# the OSCC and the LSM
from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from flask_jsonpify import jsonify

# import jsonpickle

from json import dumps

# INSTALL INSTRUCTIONS
# pipInstall flask flask_cors flask_restful flask_jsonpify

# RUN INSTRUCTIONS
# FLASK_APP=<filename>.py FLASK_ENV=development flask run --port 5002

# from tensorflow.keras import models
import numpy as np
import pickle
import configparser
import pandas 

import utilities as util

from flair.models import SequenceTagger
from flair.data import Sentence

app = Flask(__name__)
api = Api(app)

CORS(app, origins = ["http://localhost:4200", "https://oscc.lucdh.nl"])

@app.route("/scan_given_lines")
def scan_given_lines():
    given_lines = request.args.get('given_lines')
    
    for line in given_lines.splitlines():
        print('hey', line, '\n')

    # load the model you trained
    model = SequenceTagger.load('experimental/resources/taggers/example-upos/final-model.pt')
    


    # create example sentence
    sentence = Sentence('ar ma vi rum que ca no troi ae qui pri mus ab or is')

    # predict tags and print
    model.predict(sentence)

    print(sentence.to_tagged_string())


    return jsonify({})

@app.route("/Get_neural_data")
def Get_neural_data():
    cf = configparser.ConfigParser()
    cf.read("config.ini")

    line_number = int(request.args.get('line_number'))
    book_number = int(request.args.get('book_number'))

    # with open('./pickle/X.npy', 'rb') as f:
        # X = np.load(f, allow_pickle=True)
    # with open('./pickle/y.npy', 'rb') as f:
        # y = np.load(f, allow_pickle=True)

    # model = models.load_model('./pickle/model')

    # # This works fine for binary classification
    # yhat = model.predict(X)

    # # Predict and test the first 10 lines. Also, print the similarity of predicted and expected
    # expected = y[line_number-1]

    df_nn = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'prediction_df'))
    df_hmm = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'hmm_prediction_df'))
    df_seqlab = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'seqlab_prediction_df'))

    df_expected = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'flattened_vectors'))

    df_expected_filtered = df_expected[(df_expected['book'] == book_number) & (df_expected['line'] == line_number)].reset_index()
    df_nn_filtered = df_nn[(df_nn['book'] == book_number) & (df_nn['line'] == line_number)].reset_index()
    df_hmm_filtered = df_hmm[(df_hmm['book'] == book_number) & (df_hmm['line'] == line_number)].reset_index()
    df_seqlab_filtered = df_seqlab[(df_seqlab['book'] == book_number) & (df_seqlab['line'] == line_number)].reset_index()

    # Overall information
    syllables = df_nn_filtered['syllable'][0]
    syllables = [i for i in syllables if i != 0] # Trim padding


    expected = df_expected_filtered['length'][0]
    expected = expected[:len(syllables)] # Trim padding

    # Neural network prediction
    predicted_nn = df_nn_filtered['predicted'][0]
    predicted_nn = predicted_nn[:len(syllables)] # Trim padding
    predicted_nn_int = [round(num) for num in predicted_nn]

    labels_nn_predicted = ['—' if i==1 else '⏑' if i==0 else i for i in predicted_nn_int]
    labels_expected = ['—' if i==1 else '⏑' if i==0 else '∅' for i in expected]

    print(expected, '\n', labels_expected)

    predicted_hmm = df_hmm_filtered['predicted'][0]
    labels_hmm_predicted = ['—' if i=='long' else '⏑' if i=='short' else '∅' for i in predicted_hmm]

    predicted_seqlab = df_seqlab_filtered['predicted'][0]
    labels_seqlab_predicted = ['—' if i=='long' else '⏑' if i=='short' else '∅' for i in predicted_seqlab]

    correct_list_nn = calculate_list_similarity(labels_expected, labels_nn_predicted)
    correct_list_hmm = calculate_list_similarity(labels_expected, labels_hmm_predicted)
    correct_list_seqlab = calculate_list_similarity(labels_expected, labels_seqlab_predicted)

    # Dirty hack for Angular
    labels_hmm_predicted.append('HMM')
    labels_nn_predicted.append('NN')
    labels_seqlab_predicted.append('SeqLab')
    labels_expected.append('Expected')
    syllables.append('Syllables')

    result = {
        "syllables" : syllables,
        "expected" : list(labels_expected),
        "nn_predicted" : list(labels_nn_predicted),
        "hmm_predicted" : list(labels_hmm_predicted),
        "seqlab_predicted" : list(labels_seqlab_predicted),
        "correct_list_nn" : correct_list_nn,
        "correct_list_hmm" : correct_list_hmm,
        "correct_list_seqlab": correct_list_seqlab,
        "length": len(syllables),
    }

    return jsonify(result)

def calculate_list_similarity(expected, predicted):

    correct_list = []

    for i in range(len(predicted)):

        if predicted[i] == expected[i]:
            correct_list.append('lightgreen')
        else:
            correct_list.append('orange')

    return correct_list

# MAIN
if __name__ == '__main__':
    context = ('/etc/letsencrypt/live/oscc.nolden.biz/cert.pem', '/etc/letsencrypt/live/oscc.nolden.biz/privkey.pem')
    app.run(host='0.0.0.0', port=5002, ssl_context = context)
