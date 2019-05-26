# %% Change working directory from the workspace root to the ipynb file location.
import os
import pandas as pd
import numpy as np
import tensorflow as tf
# %%
class Reader(object):
    def __init__(self, feature_extraction_method, data_file, test_file, batch_size, validation_split, variance_threshold):
        self._data_file = data_file
        self._test_file = test_file
        self._batch_size = batch_size
        self.validation_split = validation_split
        self.variance_threshold = variance_threshold
        self.feature_extraction_method = feature_extraction_method

    def import_data(self):
        """ READS DATA FROM INPUT FILES AND SEPPARATES FEATURES FROM TARGETS
            Also feature named 'md5' gets dropped because it has
            too large values to be computed.

            Class data members: 
            self.inputs, self.targets - resulting training inputs and targets
            self.test_inputs, self.test_targets - resulting testing inputs and targets 
        """
        print("---------- Importing data from file")
        df = pd.read_csv(self._data_file, low_memory=False)
        df_test = pd.read_csv(self._test_file, low_memory=False)

        df = df.reindex(np.random.permutation(df.index))
        
        self.inputs = df.drop(labels=['verdict', 'md5'], axis='columns')
        self.test_inputs = df_test.drop(labels=['md5'], axis='columns')

        self.inputs.replace(to_replace={True: 1, False: 0}, inplace=True)
        self.test_inputs.replace(to_replace={True: 1, False: 0}, inplace=True)        

        self.targets = df['verdict']
        self.targets.replace(to_replace={'trojan': 1, 'clean': 0}, inplace=True)

        # self.test_targets = df_test['verdict']
        # self.test_targets.replace(to_replace={'trojan': 1, 'clean': 0}, inplace=True)
        self.test_targets = np.ones(self.test_inputs.shape[0]).astype(int)
       

        self.inputs = np.array(self.inputs)
        self.test_inputs = np.array(self.test_inputs)
        self.targets = np.array(self.targets)
        self.test_targets = np.array(self.test_targets)

    def _remove_zero_stdev_features(self):
        """ DELETES FEATURES WITH ZERO STANDARD DEVIATION
            Class data members:
            self.stdev - vector containing standar deviations calculated on training data for each feature
            self.mean - vector containing means calculated on training data for each feature
            self.inputs - features used in training fase
            self.test_inputs - testing features
        """
        print("---------- Drop zero variance features")
        zero_variance_features = np.where(self.stdev == 0)
        self.stdev = np.delete(self.stdev, zero_variance_features)
        self.mean = np.delete(self.mean, zero_variance_features)
        self.inputs = np.delete(self.inputs, zero_variance_features, axis=1)
        self.test_inputs = np.delete(self.test_inputs, zero_variance_features, axis=1)

    def scale_data(self):
        """ SCALE DATA TO 0 MEAN AND 1 STDV
            Class data members:
            self.stdev - vector containing standar deviations calculated on training data for each feature
            self.mean - vector containing means calculated on training data for each feature
            self.inputs - features used in training fase
            self.test_inputs - testing features
        """
        print("---------- Scale features")
        self.mean = np.mean(self.inputs, axis=0)
        self.stdev = np.std(self.inputs, axis=0)
        self._remove_zero_stdev_features()
        self.inputs = (self.inputs - self.mean) / self.stdev
        self.test_inputs = (self.test_inputs - self.mean) / self.stdev

    def split_train_validation_test(self):
        """ SPLIT INPUTS INTO TRAIN FEATURES AND VALIDATION FEATURES
            Class data members:
            self.train_inputs - resulting training features
            self.validation_inputs - resulting validation features
        """
        print("---------- Split data set into train, validation")
        self.train_inputs = self.inputs[0: - (int)(self.validation_split * (int)(self.inputs.shape[0]) + 1)]
        self.validation_inputs = self.inputs[- (int)(self.validation_split * (int)(self.inputs.shape[0])):]

        self.validation_targets = self.targets[- (int)(self.validation_split * self.targets.shape[0]) : ]
        self.train_targets = self.targets[0: - (int)(self.validation_split * self.targets.shape[0] + 1)]

    def pca_fit(self, data, variance_threshold, session):
        """ CALCULATE SVD MATRICIES AND OBTAIN SINGULAR VALUES AND THE V MATRIX
            Args:
            data - data matrix on which SVD decomposition is calculated
            variance_threshold - 
            session - tensorflow session for used to compute the matricies and not their tensors 
            Class data members:
            self.dimensions = vector representing the eigen values with a variance larger then the requested threshold
            self.v_matrix = the V matrix representing the eigen vectors of A^T * A
        """
        print("---------- PCA fitting")
        singular_values, _, v_matrix = session.run(tf.svd(data))

        singular_values = np.square(singular_values)
        sum_sv = np.sum(singular_values)
        singular_values = singular_values / sum_sv

        singular_values = np.cumsum(singular_values)
        self.dimensions = np.where(singular_values > variance_threshold)[0][0]
        self.v_matrix = v_matrix[:, 0:self.dimensions]

    def pca_transform(self, data, session):
        """ TRANSFORM MAP DATA MATRIX TO THE NEW FEATURES SPACE CALCULATED IN pca_fit (usually having smaller dimension)
            Args:
            data - data matrix to be transformed
            Return:
            The new transformed matrix
        """
        return tf.matmul(data, self.v_matrix)

    def autoencoder_model(self, hidden_layer_1_dimension, hidden_layer_2_dimension):
        """ CREATE AUTOENCODER MODEL TO IMITATE THE FUNCTIONALITY OF PCA
            Args:
            hidden_layer_1_dimension - dimension for first hidden layer
            hidden_layer_2_dimension - dimension for second hidden layer
        """
        self.autoencoder_input = tf.placeholder(tf.float64, [None, self.train_inputs.shape[1]])
        encoder_layer_1 = tf.layers.dense(self.autoencoder_input , int(self.autoencoder_input.shape[1]), activation=tf.nn.elu, name="First_Encoder_Layer" )
        encoder_layer_2 = tf.layers.dense(encoder_layer_1, hidden_layer_1_dimension, activation=tf.nn.elu, name="Second_Encoder_Layer")
        self.encoder_output = tf.layers.dense(encoder_layer_2, hidden_layer_2_dimension, activation=tf.nn.elu, name="Output_Encoder_Layer" )

        decoder_layer_1 = tf.layers.dense(self.encoder_output, hidden_layer_1_dimension, activation=tf.nn.elu, name="First_Decoder_Layer")
        self.decoder_output = tf.layers.dense(decoder_layer_1, self.autoencoder_input.shape[1], activation=tf.nn.sigmoid, name="Output_Decoder_Layer" )

        reconstruction_loss = tf.reduce_mean(tf.square(self.decoder_output - self.autoencoder_input))
        self.autoencoder_optimizer = tf.train.AdamOptimizer(0.003).minimize(reconstruction_loss)

    def autoencoder_fit(self, epochs, session):
        """ FIT THE AUTOENCODER MODEL TO THE TRAINING INPUTS
            Args :
            epochs - number of epochs used for training the autoencoder
        """
        print("---------- Autoencoder fitting")
        session.run(tf.global_variables_initializer())
        for _ in range(epochs):
            session.run(self.autoencoder_optimizer, feed_dict = {self.autoencoder_input: self.train_inputs})

    def autoencoder_transform(self, data, session):
        """ TRANSFORM DATA INPUT REDUCING DIMENSIONALITY BY CALCULATING THE OUTPUT OF ONY THE ENCODER PART OF THE AUTOENCODER
            Args :
            data - data to be transformed
        """
        return session.run(self.encoder_output, feed_dict={self.autoencoder_input: data})

    def compute_batches(self, data, batch_nr, batch_size, target):
        """ REORGANIZE DATA ARGUMENT IN BATCHES
        Args :
            data - data to be transformed
            batch_nr - new number of raws representing the number of batches
            batch_size - number of sambles per batch
            target - boolean value representing if the data is a matrix of features or targets
        """
        if not target:
            return tf.cast(tf.reshape(data[0: (int)(batch_nr * batch_size)], ((batch_nr, batch_size, data.shape[1]))), tf.float64)
        else:
            return tf.cast(tf.reshape(data[0: (int)(batch_nr * batch_size)], ((batch_nr, batch_size, 1))), tf.float64)

    def create_queue(self, session, X, Y,  batch_nr):
        """ CREATE TENSORFLOW QUEUES THAT WILL PROVIDE THE DATA ONE BATCH AT A TIME
            Args :
            X - features to extract batch from
            Y - targets to extract batch from
            batch_nr - number of batches
            Return:
            X[i], Y[i] - features and targets batch
            qr - queue runner used used for parallelism
        """
        with tf.name_scope("Input_Queues"):
            queue = tf.FIFOQueue(capacity=batch_nr / 2, dtypes=[tf.int32])
            enqueue_op = queue.enqueue_many([[j for j in range(batch_nr)]])
            i = queue.dequeue()
            qr = tf.train.QueueRunner(
                queue=queue, enqueue_ops=[enqueue_op] * 2)
        return X[i], Y[i], qr

    def batch_producer(self, session, name=None):
        """
            This method calls all the above defined methods in order to compute 
            the necessary features and targets for training, validaiton and 
            testing. 
            User is able to use PCA or Autoencoders based on the choosed config
            from the Challange model.
        """
        print("---------- Produce data batches")
        self.import_data()
        self.scale_data()
        self.split_train_validation_test()

        if self.feature_extraction_method == 'PCA':
            self.pca_fit(self.train_inputs, self.variance_threshold, session)
            print("---------- PCA transformation")
            self.train_inputs = self.pca_transform(self.train_inputs, session)
            self.validation_inputs = self.pca_transform(self.validation_inputs, session)
            self.test_inputs = self.pca_transform(self.test_inputs, session)
        else:
            self.autoencoder_model(195,160)
            self.autoencoder_fit(5, session)
            self.train_inputs = self.autoencoder_transform(self.train_inputs, session)
            self.validation_inputs = self.autoencoder_transform(self.validation_inputs, session)
            self.test_inputs = self.autoencoder_transform(self.test_inputs, session)

        self.train_inputs = tf.convert_to_tensor(self.train_inputs)
        self.validation_inputs = tf.convert_to_tensor(self.validation_inputs)
        self.test_inputs = tf.convert_to_tensor(self.test_inputs)

        train_batch_nr = (int)(self.train_inputs.shape[0]) // self._batch_size
        validation_batch_nr = 10
        validation_batch_size = self.validation_inputs.shape[0] // validation_batch_nr

        self.train_inputs = (self.compute_batches((self.train_inputs),
                                                  train_batch_nr,
                                                  self._batch_size,
                                                  False))
        self.train_targets = (self.compute_batches((self.train_targets),
                                                   train_batch_nr,
                                                   self._batch_size,
                                                   True))
        self.validation_inputs = (self.compute_batches((self.validation_inputs),
                                                       validation_batch_nr,
                                                       validation_batch_size,
                                                       False))
        self.validation_targets = (self.compute_batches((self.validation_targets),
                                                        validation_batch_nr,
                                                        validation_batch_size,
                                                        True))
        
        with tf.name_scope("BitDefender_Inputs_Train_Generator"):
            X_train, y_train, qr_train = self.create_queue(session, self.train_inputs, self.train_targets, train_batch_nr)

        with tf.name_scope("BitDefender_Inputs_Validation_Generator"):
            X_validation, y_validation, qr_validation = self.create_queue(session, self.validation_inputs, self.validation_targets, validation_batch_nr)

        with tf.name_scope("BitDefender_Inputs_Test_Generator"):
            self.test_inputs =  tf.cast(tf.convert_to_tensor(self.test_inputs), tf.float64)
            self.test_targets = tf.cast(tf.reshape(tf.convert_to_tensor(self.test_targets), (self.test_targets.shape[0],1)), tf.float64)
            (X_test, y_test) = self.test_inputs, self.test_targets

        self._coord = tf.train.Coordinator()
        self._threads_train = qr_train.create_threads(session, self._coord, start=True)
        self._threads_validation = qr_validation.create_threads(session, self._coord, start=True)

        return (X_train, y_train), (X_validation, y_validation), (X_test, y_test), (train_batch_nr, validation_batch_nr, 1)

    def free_threads(self):
        """ FREESES THE THREADS CREATED BY THE QUEUE RUNNER
        """
        self._coord.request_stop()
        self._coord.join(self._threads_train)
        self._coord.join(self._threads_validation)

#%%


#%%
