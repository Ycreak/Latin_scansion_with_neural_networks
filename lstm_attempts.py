            exit(0)
            ###

            base_model = Model(input_layer, model)

            from tensorflow_addons.text.crf_wrapper import CRFModelWrapper

            model = CRFModelWrapper(base_model, num_labels)
            model.compile(optimizer=tf.keras.optimizers.Adam(0.02))
            model.fit(X, epochs=10, verbose=2)

            exit(0)
            # from tf2crf import CRF, ModelWithCRFLoss

            crf = CRF(units=5)
            output = crf(output_layer)
            model = ModelWithCRFLoss(base_model, sparse_target=True)
            model.compile(optimizer='adam')

            # x = [[5, 2, 3] * 3] * 10
            # y = [[1, 2, 3] * 3] * 10

            model.fit(x=X, y=y, epochs=2, batch_size=32)
            # model.save('tests/1')

            exit(0)

            crf_layer = tfa.layers.CRF(num_labels, type='float32')  # CRF layer
            output_layer = crf_layer(output_layer)
            
            # decoded_sequence, potentials, sequence_length, chain_kernel = crf(kernel)  # output
            # crf_layer = CRF(units=TAG_COUNT)
            # output_layer = crf_layer(model)

            model = Model(input_layer, output_layer)
            
            model.compile(
                # optimizer="rmsprop",
                optimizer=keras.optimizers.RMSprop(),  # Optimizer
                # Loss function to minimize
                # loss='categorical_crossentropy',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                # loss=tfa.losses.SigmoidFocalCrossEntropy(),
                # List of metrics to monitor
                # metrics=[keras.metrics.SparseCategoricalAccuracy()],
                metrics=tf.keras.metrics.CategoricalAccuracy(),
            )
            
            
            # model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=["accuracy"])
            
            
            
            # model.add_loss(tf.abs(tf.reduce_mean(kernel)))
            history = model.fit(X, y, batch_size=32, epochs=epochs, validation_split=0.1, verbose=self.FLAGS.verbose)

            exit(0)
            return model



            crf_layer = CRF(num_labels)
            output_layer = crf_layer(model)

            model = Model(input_layer, output_layer)

            model.add_loss(lambda: tf.abs(tf.reduce_mean(X)))
            model.add_metric(tfa.losses.sigmoid_focal_crossentropy(X, y), name='f1_luukie')

            print(len(model.losses))



            adam = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
            #model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
            model.compile(optimizer="rmsprop")

            exit(0)

            # Compile the model
            loss = tf.keras.losses.CategoricalCrossentropy() #tfa.losses.SigmoidFocalCrossEntropy() #"categorical_crossentropy" #losses.crf_loss

            acc_metric = tfa.metrics.F1Score #crf_accuracy
            opt = "rmsprop" #optimizers.Adam(lr=0.001)

            model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])            
            history = model.fit(X, y, batch_size=32, epochs=epochs, validation_split=0.1, verbose=self.FLAGS.verbose)
            
            # y_pred = model.predict(x_train)
            # predictions = tf.argmax(y_pred,axis=1)


            # model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
            # history = model.fit(X, y, batch_size=32, epochs=epochs, verbose=self.FLAGS.verbose)