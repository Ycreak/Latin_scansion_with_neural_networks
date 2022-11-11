            def run_idx_lstm_single_text(self, text, 
                                 do_evaluate=True, do_metric_report=True, do_confusion_matrix=True, print_stats=True):        
        # Retrieve meta data
        all_text_syllables = self.retrieve_syllables_from_sequence_label_list(text)
        max_sentence_length = self.retrieve_max_sentence_length(text)
        unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING) # needs to be sorted for word2idx consistency!

        # # Create dictionary
        word2idx = {w: i for i, w in enumerate(unique_syllables)}
        label2idx = {t: i for i, t in enumerate(self.LABELS)}
        # now we map the sentences and labels to a sequence of numbers
        X = [[word2idx[w[0]] for w in s] for s in text]  # key 0 are labels
        y = [[label2idx[w[1]] for w in s] for s in text] # key 1 are labels
        # and then (post)pad the sequences using the PADDING label.
        X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=word2idx[self.PADDING]) # value is our padding key
        y = pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=label2idx[self.PADDING])
        # for training the network we also need to change the labels to categorial.
        y = np.array([to_categorical(i, num_classes=len(self.LABELS)) for i in y])
        # we split in train and test set.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split_size)
        # get the model we want
        model = self.get_model( max_len = max_sentence_length,
                                num_syllables = len(unique_syllables),
                                num_labels = len(self.LABELS),
                                X = X_train,
                                y = y_train,
                                epochs = FLAGS.epochs,
                                create_model = FLAGS.create_model,
                                save_model = FLAGS.save_model)
        
        if do_evaluate:
            loss, accuracy = self.evaluate_model(model, X_test, y_test)
            result_string = 'RESULT: loss -> {0}, accuracy -> {1}. Length of test set: {2}\n'.format(loss, accuracy, len(y_test))
            print(result_string)
            # return loss, accuracy, len(y_train), len(y_test)

        if do_metric_report:
            metrics_report = self.create_metrics_report(model, X_test, y_test, output_dict=False)
            print(metrics_report)

        if do_confusion_matrix:
            confusion_matrix = self.create_confusion_matrix(model, X_test, y_test)
            df_confusion_matrix = pd.DataFrame(confusion_matrix, index = ['long', 'short', 'elision', 'space', 'padding'],
                                            columns = ['long', 'short', 'elision', 'space', 'padding'])
            # Drop the padding labels, as we don't need them (lstm scans them without confusion): delete both row and column
            df_confusion_matrix = df_confusion_matrix.drop('padding')
            df_confusion_matrix = df_confusion_matrix.drop(columns=['padding', 'space'])

            df_confusion_matrix = df_confusion_matrix.drop('space')
            # df_confusion_matrix = df_confusion_matrix.drop(columns=['padding'])

            util.create_heatmap(dataframe = df_confusion_matrix,
                                xlabel = 'TRUTH', 
                                ylabel =  'PREDICTED',
                                title = 'CONFUSION MATRIX',
                                filename = 'confusion_matrix_lstm_single_text',
                                # vmax = 500
                                ) 
        #################
        # TEMP PRINTING #
        #################
        if print_stats:
            # Keep track of wrong scansions
            sentence_incorrect_counter = 0
            syllable_incorrect_counter = 0
            
            # X = X_test
            # y = y_test

            # from collections import Counter
            # result = Counter(x for xs in y for x in set(xs))
            # print('RESULT', result)

            # Make a prediction over the whole dataset
            y_pred = model.predict(X)
            y_pred = np.argmax(y_pred, axis=-1)
            
            for idx, line in enumerate(X):
                # for every line in X, create a prediction and truth
                sentence = X[idx]          
                y_pred_current = y_pred[idx]
                y_true_current = np.argmax(y[idx], axis=-1)

                # If a line is not perfectly scanned, lets investigate
                if not (y_pred_current == y_true_current).all():
                    # count the syllables for every line that are not scanned correctly
                    syllable_incorrect_counter += (max_sentence_length - np.count_nonzero(y_pred_current==y_true_current))
                    # retrieve the syllables from the hashes so we can read the sentence again 11
                    syllable_list = []
                    for item in sentence:
                        syllable_list.append(list(word2idx.keys())[list(word2idx.values()).index(item)])                  
                    # now, for each line, print some information to investigate the problems
                    while 'PADDING' in syllable_list: syllable_list.remove('PADDING')    
                    # while 'PADDING' in syllable_list: syllable_list.remove('PADDING')    


                    print('syllables : ', syllable_list)
                    # print('prediction: ', y_pred_current)
                    # print('truth:      ', y_true_current)
                    # also, for latinists, print the scansion
                    y_pred_labels = ['—' if j==0 else '⏑' if j==1 else 'e' if j==2 else ' ' if j==3 else j for j in y_pred_current]
                    y_train_labels = ['—' if j==0 else '⏑' if j==1 else 'e' if j==2 else ' ' if j==3 else j for j in y_true_current]

                    while 4 in y_pred_labels: y_pred_labels.remove(4)    # padding.
                    while 4 in y_train_labels: y_train_labels.remove(4)    

                    print('prediction: ', y_pred_labels)
                    print('truth     : ', y_train_labels)
                    
                    print('\n##########################\n')
                    # count the sentence as incorrect
                    sentence_incorrect_counter += 1
                    


            # after all scrutiny, print the final statistics                    
            score_sentences = round(sentence_incorrect_counter/len(X)*100,2)
            # score_syllables = round(syllable_incorrect_counter/len(all_text_syllables)*100,2)

            print('SENTENCES SCANNED WRONGLY: ', sentence_incorrect_counter)
            print('PERCENTAGE WRONG: {0}%'.format(score_sentences))
            
            print('SYLLABLES SCANNED WRONGLY: ', syllable_incorrect_counter)
            # print('PERCENTAGE WRONG: {0}%'.format(score_syllables)) # This also counts spaces: needs fixing

        
        
        if FLAGS.exp_hexameter:

            train_texts = ['VERG-aene.pickle', 'IVV-satu.pickle', 'LVCR-rena.pickle', 'OV-meta.pickle', 'PERS-satu.pickle']
            test_texts = ['VERG-aene.pickle', 'IVV-satu.pickle', 'LVCR-rena.pickle', 'OV-meta.pickle', 'PERS-satu.pickle']

            # train_texts = ['VERG-aene.pickle', 'PROP-ele.pickle', 'OV-ele.pickle']
            # test_texts = ['VERG-aene.pickle', 'PROP-ele.pickle', 'OV-ele.pickle']

            # train_texts = ['PERS-satu.pickle','IVV-satu.pickle']
            # test_texts = ['PERS-satu.pickle','IVV-satu.pickle']

            sequence_labels_all_set = util.merge_sequence_label_lists(train_texts, util.cf.get('Pickle', 'path_sequence_labels')) # Merge them into one big file list
            all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
            # We need to extract the max sentence length over all these texts to get the padding correct later
            max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)       
            # And we need to create a list of all unique syllables for our word2idx one-hot encoding
            unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING)
            word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)  
            # twice test texts because they are identical and merge_sequence_label_lists has a bug
            self.do_experiment(test_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='hexameter', plot_title='Cross author evaluation')

        if FLAGS.exp_transfer_boeth:

            # To make the LSTM with integer hashing working, we need to make a list of all syllables from all the texts we are looking at    
            all_sequence_label_pickles = util.Create_files_list(util.cf.get('Pickle', 'path_sequence_labels'), 'pickle') # Find all pickle files
            sequence_labels_all_set = util.merge_sequence_label_lists(all_sequence_label_pickles, util.cf.get('Pickle', 'path_sequence_labels')) # Merge them into one big file list
            # all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
            # # We need to extract the max sentence length over all these texts to get the padding correct later
            # max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)       
            # text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), 'HEX_ELE-all.pickle')
            all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
            max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)
            # unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING) # needs to be sorted for word2idx consistency!

            # And we need to create a list of all unique syllables for our word2idx one-hot encoding
            unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING)
            word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)

            train_texts = ['VERG-aene.pickle', 'HEX-all.pickle', 'ELE-all.pickle', 'HEX_ELE-all.pickle']
            test_texts = ['BOETH-cons.pickle', 'TIB-ele.pickle', 'ENN-anna.pickle', 'HOR-arpo.pickle',
                          'LVCAN-phar.pickle', 'CATVLL-carm.pickle', 'STAT-theb.pickle']          
            
            self.do_experiment(train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='boethius', plot_title='Scanning Unseen Texts')

        if FLAGS.exp_transfer:
            # Here we test whether training on elegiac and hexameter gives better results
            train_texts = ['VERG-aene.pickle', 'HEX-all.pickle', 'ELE-all.pickle', 'HEX_ELE-all.pickle']
            test_texts = ['SEN-aga.pickle']
            self.do_experiment(train_texts, test_texts, max_sentence_length, unique_syllables, word2idx, label2idx, exp_name='seneca', plot_title='Scanning Iambic Trimeter')

        if FLAGS.exp_elegiac:
            # Pick all the elegiac texts and let Virgil do his best :D
            train_texts = ['VERG-aene.pickle', 'TIB-ele.pickle', 'PROP-ele.pickle', 'OV-ele.pickle']
            test_texts = ['VERG-aene.pickle','TIB-ele.pickle', 'PROP-ele.pickle', 'OV-ele.pickle']
            self.do_experiment(train_texts, test_texts, exp_name='elegiac')

        if FLAGS.exp_train_test:
            train_texts = ['VERG-aene.pickle', 'IVV-satu.pickle', 'LVCR-rena.pickle', 'OV-meta.pickle']
            # train_texts = ['VERG-aene.pickle']

            for text in train_texts:
                current_text = util.Pickle_read(util.cf.get('Pickle', 'path_sequence_labels'), text)
                current_text = random.sample(current_text, 3600)

                all_text_syllables = self.retrieve_syllables_from_sequence_label_list(current_text)
                max_sentence_length = self.retrieve_max_sentence_length(current_text)
                unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING) # needs to be sorted for word2idx consistency!

                word2idx, label2idx = self.create_idx_dictionaries(unique_syllables, self.LABELS)

                X, y = self.create_X_y_sets(current_text, word2idx, label2idx, max_sentence_length)
             
                # we split in train and test set.
                X_test = X[:720]    # 20% of 3600
                y_test = y[:720]
                X_train = X[720:]
                y_train = y[720:]
                
                splits = 30

                X_train_list = np.array_split(X_train, splits)
                y_train_list = np.array_split(y_train, splits)

                X_train_list_2 = X_train_list[splits-1]
                y_train_list_2 = y_train_list[splits-1]

                # print(X_train_list[1])

                for i in range(splits):

                    # get the model we want
                    model = self.get_model( max_len = max_sentence_length,
                                            num_syllables = len(unique_syllables),
                                            num_labels = len(self.LABELS),
                                            X = X_train_list_2,
                                            y = y_train_list_2,
                                            epochs = FLAGS.epochs,
                                            create_model = True,
                                            save_model = False)
                    
                    loss, accuracy = self.evaluate_model(model, X_test, y_test)

                    result = '{0},{1},{2}\n'.format(text, accuracy, len(X_train_list_2))
                    # Open a file with access mode 'a'

                    file_name = './plots/size_' + text + '.txt'
                    file_object = open(file_name, 'a')
                    # Append 'hello' at the end of file
                    file_object.write(result)
                    # Close the file
                    file_object.close()

                    # Now increase the train list and repeat the experiment
                    X_train_list_2 = np.append(X_train_list_2, X_train_list[i], axis=0)
                    y_train_list_2 = np.append(y_train_list_2, y_train_list[i], axis=0)