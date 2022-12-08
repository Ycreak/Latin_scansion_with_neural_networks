    ###############
    # CRF EXPERIMENTS #
    ###############
    def elegiac_heatmap(self):

        column_names = ["predictor", "predictee", "score"]
        df_long = pd.DataFrame(columns = column_names)
        df_short = pd.DataFrame(columns = column_names)
        df_elision = pd.DataFrame(columns = column_names)

        ovid_elegiac_df = util.merge_sequence_label_lists(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), 'OV'), util.cf.get('Pickle', 'df_crf_path_elegiac'))
        tib_elegiac_df = util.merge_sequence_label_lists(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), 'TIB'), util.cf.get('Pickle', 'df_crf_path_elegiac'))
        prop_elegiac_df = util.merge_sequence_label_lists(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), 'PROP'), util.cf.get('Pickle', 'df_crf_path_elegiac'))

        verg_df = util.Pickle_read(util.cf.get('Pickle', 'df_crf_path_hexameter'), 'VERG-aene.pickle')
        hexameter_df = util.merge_sequence_label_lists(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_hexameter'), '.pickle'), util.cf.get('Pickle', 'df_crf_path_hexameter'))
        elegiac_df = util.merge_sequence_label_lists(util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_elegiac'), '.pickle'), util.cf.get('Pickle', 'df_crf_path_elegiac'))
        
        # First, use Aneid to predict Ovid and Tibullus
        predictor_texts = [('verg',verg_df), ('hex',hexameter_df), ('eleg',elegiac_df)]
        predictee_texts = [('ovid',ovid_elegiac_df), ('tib',tib_elegiac_df), ('prop',prop_elegiac_df)]

        for predictor_text in predictor_texts:
            predictor_X, predictor_y = self.convert_text_to_feature_sets(predictor_text[1])
            crf_model = self.fit_model(predictor_X, predictor_y)

            for predictee_text in predictee_texts:
                predictee_X, predictee_y = self.convert_text_to_feature_sets(predictee_text[1])
                # Using the predictor model, predict for the predictee test set
                result = self.predict_model(crf_model, predictee_X, predictee_y)

                predictor = predictor_text[0] # get names
                predictee = predictee_text[0]

                print(predictor, predictee)
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

        # pivot the dataframe to be usable with the seaborn heatmap
        heatmap_data = pd.pivot_table(df_long, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- long syllables',
                        filename = 'confusionmatrix_elegiac_long.png')

        heatmap_data = pd.pivot_table(df_short, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- short syllables',
                        filename = 'confusionmatrix_elegiac_short.png')

        heatmap_data = pd.pivot_table(df_elision, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- elided syllables',
                        filename = 'confusionmatrix_elegiac_elision.png')
                        
        # exit(0)


    def hexameter_heatmap(self):

        ############################
        # CREATE HEXAMETER HEATMAP #
        ############################
        column_names = ["predictor", "predictee", "score"]
        df_long = pd.DataFrame(columns = column_names)
        df_short = pd.DataFrame(columns = column_names)
        df_elision = pd.DataFrame(columns = column_names)

        predictor_texts = util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_hexameter'), '.pickle')
        predictee_texts = util.Create_files_list(util.cf.get('Pickle', 'df_crf_path_hexameter'), '.pickle')

        # Let every text be the predictor once
        for predictor_text in predictor_texts:
            predictor_df = util.Pickle_read(util.cf.get('Pickle', 'df_crf_path_hexameter'), predictor_text)
            predictor_X, predictor_y = self.convert_text_to_feature_sets(predictor_df)
            crf_model = self.fit_model(predictor_X, predictor_y)

            # For every predictor, get results for every predictee. This includes predicting itself
            for predictee_text in predictee_texts:
                predictee_df = util.Pickle_read(util.cf.get('Pickle', 'df_crf_path_hexameter'), predictee_text)
                predictee_X, predictee_y = self.convert_text_to_feature_sets(predictee_df)
                # Using the predictor model, predict for the predictee test set
                result = self.predict_model(crf_model, predictee_X, predictee_y)

                predictor = predictor_text.split('-')[0].capitalize() # get names
                predictee = predictee_text.split('-')[0].capitalize()
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

        # df_long.to_csv('./csv/long_1_char_only.csv')
        # df_short.to_csv('./csv/short_1_char_only.csv')
        # df_elision.to_csv('./csv/elision_1_char_only.csv')
        
        # pivot the dataframe to be usable with the seaborn heatmap
        heatmap_data = pd.pivot_table(df_long, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- long syllables',
                        filename = 'confusionmatrix_hexameter_long.png')

        heatmap_data = pd.pivot_table(df_short, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- short syllables',
                        filename = 'confusionmatrix_hexameter_short.png')


        heatmap_data = pd.pivot_table(df_elision, values='score', index=['predictor'], columns='predictee')
        self.create_heatmap(dataframe = heatmap_data,
                        xlabel = 'predictee',
                        ylabel = 'predictor',
                        title = 'Confusion matrix -- elided syllables',
                        filename = 'confusionmatrix_hexameter_elision.png')

    def create_heatmap(self, dataframe, xlabel, ylabel, title, filename):
        # Simple function to create a heatmap
        sn.set(font_scale=1.4)
        sn.heatmap(dataframe, annot=True, fmt='g', annot_kws={"size": 16}, cmap='Blues')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(filename, bbox_inches='tight')        
        plt.clf()    
