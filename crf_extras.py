import pandas as pd
import utilities as util
import seaborn as sn
import matplotlib.pyplot as plt


    def run_experiments(self):
        create_models = True
        
        crf_exp_bar_dict = {'exp1': {}, 'exp2': {}, 'exp3': {}, 'exp4': {}, 'exp5': {}}

        # Experiment 1: Create model on Virgil, test on Virgil
        if create_models:
            texts = ['syllable_label_VERG-aene.xml.pickle']
            crf_df = self.convert_pedecerto_to_crf_df(texts)
            X, y = self.convert_text_to_feature_sets(crf_df)
            util.Pickle_write(util.cf.get('Pickle', 'path'), 'crf_exp1_X.pickle', X)
            util.Pickle_write(util.cf.get('Pickle', 'path'), 'crf_exp1_y.pickle', y)
        else:
            X = util.Pickle_read(util.cf.get('Pickle', 'path'), 'crf_exp1_X.pickle')
            y = util.Pickle_read(util.cf.get('Pickle', 'path'), 'crf_exp1_y.pickle')
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        crf_model = self.fit_model(X_train, y_train)
        result = self.predict_model(crf_model, X_test, y_test)
        print('exp1')
       
        crf_exp_bar_dict['exp1'] = {'short_precision': result['short']['precision'],
                                    'short_recall': result['short']['recall'],
                                    'long_precision': result['long']['precision'],
                                    'long_recall': result['long']['recall'],
                                    'elision_precision': result['elision']['precision'],
                                    'elision_recall': result['elision']['recall'],
                                   }

        # Experiment 4: Test Virgil on Hercules Furens
        herc_df = pd.read_csv('HercFur.csv')  
        util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'test'), herc_df)
        herc_df = self.convert_pedecerto_to_crf_df(['test.pickle'])
        X_herc, y_herc = self.convert_text_to_feature_sets(herc_df)
        result = self.predict_model(crf_model, X_herc, y_herc)
        print('exp4')
        
        crf_exp_bar_dict['exp4'] = {'short_precision': result['short']['precision'],
                                    'short_recall': result['short']['recall'],
                                    'long_precision': result['long']['precision'],
                                    'long_recall': result['long']['recall'],
                                    'elision_precision': result['elision']['precision'],
                                    'elision_recall': result['elision']['recall'],
                                   }
        

        # Create model on Virgil, Ovid, Iuvenal and Lucretius, test on Aeneid
        texts = util.Create_files_list('./pickle', 'syllable_label')
        crf_df = self.convert_pedecerto_to_crf_df(texts)
        X, y = self.convert_text_to_feature_sets(crf_df)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        crf_model = self.fit_model(X_train, y_train)
        result = self.predict_model(crf_model, X_test, y_test)
        print('exp2')

        crf_exp_bar_dict['exp2'] = {'short_precision': result['short']['precision'],
                                    'short_recall': result['short']['recall'],
                                    'long_precision': result['long']['precision'],
                                    'long_recall': result['long']['recall'],
                                    'elision_precision': result['elision']['precision'],
                                    'elision_recall': result['elision']['recall'],
                                   }

        # Create model on Virgil, Ovid, Iuvenal and Lucreatius, test on all
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        crf_model = self.fit_model(X_train, y_train)
        result = self.predict_model(crf_model, X_test, y_test)
        print('exp3')
        
        crf_exp_bar_dict['exp3'] = {'short_precision': result['short']['precision'],
                                    'short_recall': result['short']['recall'],
                                    'long_precision': result['long']['precision'],
                                    'long_recall': result['long']['recall'],
                                    'elision_precision': result['elision']['precision'],
                                    'elision_recall': result['elision']['recall'],
                                   }

        util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'test'), crf_exp_bar_dict)

        # herc_df = pd.read_csv('HercFur.csv')  
        # crf_df = self.convert_pedecerto_to_crf_df(['test.pickle'])
        # X, y = self.convert_text_to_feature_sets(herc_df)
        result = self.predict_model(crf_model, X_herc, y_herc)
        print('exp5')
        crf_exp_bar_dict['exp5'] = {'short_precision': result['short']['precision'],
                                    'short_recall': result['short']['recall'],
                                    'long_precision': result['long']['precision'],
                                    'long_recall': result['long']['recall'],
                                    'elision_precision': result['elision']['precision'],
                                    'elision_recall': result['elision']['recall'],
                                   }

        pd.DataFrame(crf_exp_bar_dict).T.plot(kind='bar')
        plt.legend(loc='lower left')
        plt.ylim([0.5, 1])
        plt.savefig('./result.png')
        plt.show()


    def predict_custom_sentence(self, crf_model, custom_sentence):
        # Turn the sentence into the format requested by the add_word_features function
        custom_sentence = custom_sentence.split()
        dummy_list = [0] * len(custom_sentence)
        combined_list = [(custom_sentence[i], dummy_list[i]) for i in range(0, len(custom_sentence))]
        custom_sentence_features = [self.add_word_features(combined_list, i) for i in range(len(combined_list))]

        # Print the confidence for every syllable
        marginals = crf_model.predict_marginals_single(custom_sentence_features)
        marginals = [{k: round(v, 4) for k, v in marginals.items()} for marginals in marginals]
        print(marginals)
        # Print the scansion for the entire sentence
        scansion = crf_model.predict_single(custom_sentence_features)
        print(scansion)
        # Below the reason for this scansion is saved
        # For every feature dictionary that is created per syllable.
        for idx, syllable_feature_dict in enumerate(custom_sentence_features):

            print('\nWe are now looking at syllable "{}".'.format(custom_sentence[idx]))
            print('I scanned it as being "{}"'.format(scansion[idx]))
            print('My confidence is {}:'.format(marginals[idx]))
            print('Below are the reasons why:')

            # Check if its features can be found in the big state_features dictionary.
            for key in syllable_feature_dict:
                my_string = str(key) + ':' + str(syllable_feature_dict[key])

                # If it is found, print the reasoning so we can examine it.
                for k, v in crf_model.state_features_.items():
                    if my_string in k:
                        print(k, v)

    def print_state_features(self, state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def str_is_consonants(self, word, slicer) -> bool:
        return all([char in self.CONSONANTS for char in word[slicer]])

    def syllable_contains_diphthong(self, syllable) -> bool:
        return any(diphthong in syllable for diphthong in self.DIPTHONGS)

    def print_crf_items(self, crf):
        print("Top positive:")
        self.print_state_features(collections.Counter(crf.state_features_).most_common(30))
        print("\nTop negative:")
        self.print_state_features(collections.Counter(crf.state_features_).most_common()[-30:])                    


    def create_prediction_df(self, X, y):
        # Creates a dataframe with predictions. Used by OSCC (for now)
        df = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'flattened_vectors'))
        crf = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_model'))

        yhat = crf.predict(X)

        column_names = ["predicted", "expected"]
        new_df = pd.DataFrame(columns = column_names)

        for i in Bar('Processing').iter(range(len(y))):
            new_line = {'expected': y[i], 'predicted': yhat[i]}
            new_df = new_df.append(new_line, ignore_index=True)

        book_line_df = df[['book','line', 'syllable']]

        prediction_df = pd.concat([book_line_df, new_df], axis=1, join='inner')

        print(prediction_df)

        util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'seqlab_prediction_df'), prediction_df)

       

      

    def grid_search(self, X, y):

        crf = self.fit_model(X, y)
        
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }
        # use the same metric for evaluation
        f1_scorer = metrics.make_scorer(metrics.flat_f1_score,
                                average='weighted', labels=self.labels)
        # search
        rs = RandomizedSearchCV(crf, params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=50,
                                scoring=f1_scorer)
        rs.fit(X, y)

        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        sorted_labels = sorted(
            self.labels,
            key=lambda name: (name[1:], name[0])
        )

        crf = rs.best_estimator_
        y_pred = crf.predict(X_test)
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))

        util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'seq_lab_rs'), rs)

    perform_pedecerto_creation = False              # This creates pedecerto dataframes
    perform_pedecerto_to_sequence_labeling_conversion = False            # This converts pedecerto dataframes to crf readable dataframes
    perform_convert_text_to_feature_sets = False    # This adds features to the crf dataframes: creates X and y
    perform_kfold = False
    perform_grid_search = False
    perform_fit_model = False
    perform_prediction_df = False
    perform_experiments = False
    custom_predict = False

        # # Read all the data we will be needing. The syllable_label_list contains a list of the used texts in [(syl, lbl), (syl,lbl), ...] format.
        # crf_df = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_df'))
        # # Load training and test set: X contains syllables and their features, y contains only scansion labels per line
        # X = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_X'))
        # y = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_y'))
        # # Load our latest CRF model
        # crf_model = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_model'))

        # if self.perform_pedecerto_creation:
        #     # Create pedecerto df files from all texts in the given folder 
        #     # parse = Pedecerto_parser(util.cf.get('Pedecerto', 'path_texts'))
        #     parse = Pedecerto_parser('./texts/elegiac/')
        #     exit(0)

        # # if self.perform_pedecerto_to_sequence_labeling_conversion:
        # #     # Convert the pedecerto dataframe to a syllable_label_list as required by the used CRF suite
        # #     texts = util.Create_files_list('./pickle/df_pedecerto', '.pickle')
            
        # #     for text in texts:
        # #         df = util.Pickle_read(util.cf.get('Pickle', 'df_pedecerto_path'), text)

        # #         # Convert the integer labels to string labels 
        # #         df = util.convert_syllable_labels(df)
        # #         crf_df = self.convert_pedecerto_to_sequence_labeling(df)
                
        # #         text_name = text.split('.')[0]
        # #         text_name = text.split('_')[-1]

        # #         util.Pickle_write(util.cf.get('Pickle', 'df_crf_path'), text_name, crf_df)

        # if self.perform_convert_text_to_feature_sets:
        #     # Takes the syllable label list and adds features to each syllable that are relevant for scansion
        #     texts = util.Create_files_list(util.cf.get('Pickle', 'df_crf_path'), '.pickle')

        #     print(texts)

        #     for text in texts:
        #         df = util.Pickle_read(util.cf.get('Pickle', 'df_crf_path'), text)
                
        #         X, y = self.convert_text_to_feature_sets(df)

        #     # exit(0)
        #     # util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_X'), X)
        #     # util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_y'), y)

        # if self.perform_fit_model:
        #     # Fit the model if needed
        #     crf_model = self.fit_model(X, y)
        #     self.print_crf_items(crf_model)
        #     util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_model'), crf_model)

        # if self.perform_kfold:
        #     # Perform kfold to check if we don't have any overfitting
        #     result = self.kfold_model(crf_df, X, y, 5)
        #     util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_kfold_result'), result)
        #     print(result)

        # if self.custom_predict:
        #     # Predict a custom sentence. NB: this has to be syllabified by the user
        #     custom_sentence = "li to ra mul tum il le et ter ris iac ta tus et al to"
        #     custom_sentence = "ar ma vi rum que ca no troi ae qui pri mus ab or is"
        #     self.predict_custom_sentence(crf_model, custom_sentence)            

        # if self.perform_grid_search:
        #     # Does what it says on the tin
        #     self.grid_search(X, y)

        # if self.perform_prediction_df:
        #     # Creates a simple prediction dataframe used by the frontend to quickly load results
        #     self.create_prediction_df(X, y)

        # if self.perform_experiments:
        #     self.run_experiments()


    def crf_improvement_heatmap(self):
        
        df = pd.read_csv('./csv/improvement_crf_long.csv')  # with consonant bools

        heatmap_data = pd.pivot_table(df, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- long syllables',
                        filename = 'confusionmatrix_improve_long.png')

        df = pd.read_csv('./csv/improvement_crf_short.csv')  # with consonant bools

        heatmap_data = pd.pivot_table(df, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- short syllables',
                        filename = 'confusionmatrix_improve_short.png')

        df = pd.read_csv('./csv/improvement_crf_elision.csv')  # with consonant bools

        heatmap_data = pd.pivot_table(df, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- elided syllables',
                        filename = 'confusionmatrix_improve_elision.png')                        

        # exit(0)

        column_names = ["predictor", "predictee", "score"]
        df_long = pd.DataFrame(columns = column_names)
        df_short = pd.DataFrame(columns = column_names)
        df_elision = pd.DataFrame(columns = column_names)

        # ovid_elegiac_df = self.merge_crf_dataframes(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), 'OV'), util.cf.get('Pickle', 'df_crf_path_elegiac'))
        # tib_elegiac_df = self.merge_crf_dataframes(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), 'TIB'), util.cf.get('Pickle', 'df_crf_path_elegiac'))
        # prop_elegiac_df = self.merge_crf_dataframes(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), 'PROP'), util.cf.get('Pickle', 'df_crf_path_elegiac'))

        verg_df = util.Pickle_read(util.cf.get('Pickle', 'df_crf_path_hexameter'), 'VERG-aene.pickle')
        # hexameter_df = self.merge_crf_dataframes(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_hexameter'), '.pickle'), util.cf.get('Pickle', 'df_crf_path_hexameter'))
        # elegiac_df = self.merge_crf_dataframes(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), '.pickle'), util.cf.get('Pickle', 'df_crf_path_elegiac'))
        
        # First, use Aneid to predict Ovid and Tibullus
        predictor_texts = [('verg',verg_df)]
        predictee_texts = util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_hexameter'), '.pickle')

        for predictor_text in predictor_texts:
            predictor_X, predictor_y = self.convert_text_to_feature_sets(predictor_text[1])
            crf_model = self.fit_model(predictor_X, predictor_y)

            for predictee_text in predictee_texts:
                predictee_df = util.Pickle_read(util.cf.get('Pickle', 'df_crf_path_hexameter'), predictee_text)
                predictee_X, predictee_y = self.convert_text_to_feature_sets(predictee_df)
                # Using the predictor model, predict for the predictee test set
                result = self.predict_model(crf_model, predictee_X, predictee_y)

                predictor = predictor_text[0] # get names
                predictee = predictee_text.split('-')[0].capitalize()
                # exit(0)

                score_long = float(round(result['long']['f1-score'] * 100,1)) # save the score
                score_short = float(round(result['short']['f1-score'] * 100,1)) # save the score
                score_elision = float(round(result['elision']['f1-score'] * 100,1)) # save the score
                
                # Add score to the dataframe for our heatmap
                new_line_long = {'predictor': predictor, 'predictee': predictee, 'score': score_long}
                new_line_short = {'predictor': predictor, 'predictee': predictee, 'score': score_short}
                new_line_elision = {'predictor': predictor, 'predictee': predictee, 'score': score_elision}

                df_long = df_long.append(new_line_long, ignore_index=True)    
                df_short = df_short.append(new_line_short, ignore_index=True)    
                df_elision = df_elision.append(new_line_elision, ignore_index=True)

        print(df_long)
        print(df_short)
        print(df_elision)

