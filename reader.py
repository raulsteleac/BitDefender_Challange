# %% Change working directory from the workspace root to the ipynb file location.
import os
# try:
#     os.chdir(os.path.join(
#         os.getcwd(), 'challange'))
#     print(os.getcwd())
# except:
#     pass
# data_file = 'challenge_train.csv'

import pandas as pd
import numpy as np
import tensorflow as tf
# %%
class Reader(object):
    def __init__(self, data_file, batch_size, validation_split, test_split, variance_threshold):
        self._data_file = data_file
        self._batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.variance_threshold = variance_threshold

    def import_data(self):
        print("---------- Importing data from file")
        df = pd.read_csv(self._data_file, low_memory=False)
        df = df.reindex(np.random.permutation(df.index))

        self.inputs = df.drop(labels=['verdict', 'md5'], axis='columns')
        self.inputs.replace(to_replace={True: 1, False: 0}, inplace=True)
        self.inputs = np.array(self.inputs)

        self.targets = df['verdict']
        self.targets.replace(to_replace={'trojan': 1, 'clean': 0}, inplace=True)
        self.targets = np.array(self.targets)

    def _remove_zero_stdev_features(self):
        print("---------- Drop zero variance features")
        zero_variance_features = np.where(self.stdev == 0)
        self.stdev = np.delete(self.stdev, zero_variance_features)
        self.mean = np.delete(self.mean, zero_variance_features)
        self.inputs = np.delete(self.inputs, zero_variance_features, axis=1)

    def scale_data(self):
        print("---------- Scale features")
        self.mean = np.mean(self.inputs, axis=0)
        self.stdev = np.std(self.inputs, axis=0)
        self._remove_zero_stdev_features()
        self.inputs = (self.inputs - self.mean) / self.stdev

    def split_train_validation_test(self):
        print("---------- Split data set into train, validation and test")
        self.train_inputs = self.inputs[0: -(int)(self.test_split * (int)(self.inputs.shape[0]) + 1)]
        self.validation_inputs = self.train_inputs[- (int)(self.validation_split * (int)(self.train_inputs.shape[0])):]
        self.train_inputs = self.train_inputs[0: - (int)(self.validation_split * (int)(self.train_inputs.shape[0]) + 1)]
        self.test_inputs = self.inputs[- (int)(self.test_split * (int)(self.inputs.shape[0])):]

        self.train_targets = self.targets[0: - (int)(self.test_split * self.targets.shape[0] + 1)]
        self.validation_targets = self.train_targets[- (int)(self.validation_split * self.train_targets.shape[0]) : ]
        self.train_targets = self.train_targets[0: - (int)(self.validation_split * self.train_targets.shape[0] + 1)]
        self.test_targets = self.targets[- (int)(self.test_split * self.targets.shape[0]) : ]

    def pca_fit(self, data, variance_threshold, session):
        print("---------- PCA fitting")
        singular_values, _, v_matrix = session.run(tf.svd(data))

        singular_values = np.square(singular_values)
        sum_sv = np.sum(singular_values)
        singular_values = singular_values / sum_sv

        singular_values = np.cumsum(singular_values)
        self.dimensions = np.where(singular_values > variance_threshold)[0][0]
        print("VARIANCE = " + str(singular_values[self.dimensions]))
        self.v_matrix = v_matrix[:, :self.dimensions]

    def pca_transform(self, data):
        return tf.matmul(data, self.v_matrix)

    def compute_batches(self, data, batch_nr, batch_size, target):
        if not target:
            return tf.cast(tf.reshape(data[0: (int)(batch_nr * batch_size)], ((batch_nr, batch_size, data.shape[1]))), tf.float64)
        else:
            return tf.cast(tf.reshape(data[0: (int)(batch_nr * batch_size)], ((batch_nr, batch_size, 1))), tf.float64)

    def create_queue(self, session, X, Y,  batch_nr):
        with tf.name_scope("Input_Queues"):
            queue = tf.FIFOQueue(capacity=batch_nr / 2, dtypes=[tf.int32])
            enqueue_op = queue.enqueue_many([[j for j in range(batch_nr)]])
            i = queue.dequeue()
            qr = tf.train.QueueRunner(
                queue=queue, enqueue_ops=[enqueue_op] * 2)
        return X[i], Y[i], qr

    def batch_producer(self, session, name=None):
        print("---------- Produce data batches")
        self.import_data()
        self.scale_data()
        self.split_train_validation_test()

        self.pca_fit(self.train_inputs, self.variance_threshold, session)
        print("---------- PCA transformation")
        self.train_inputs = self.pca_transform(self.train_inputs)
        self.validation_inputs = self.pca_transform(self.validation_inputs)
        self.test_inputs = self.pca_transform(self.test_inputs)

        train_batch_nr = (int)(self.train_inputs.shape[0]) // self._batch_size
        validation_batch_nr = 10
        validation_batch_size = self.validation_inputs.shape[0] // validation_batch_nr

        self.train_inputs = (self.compute_batches(tf.convert_to_tensor(self.train_inputs),
                                                  train_batch_nr,
                                                  self._batch_size,
                                                  False))
        self.train_targets = (self.compute_batches(tf.convert_to_tensor(self.train_targets),
                                                   train_batch_nr,
                                                   self._batch_size,
                                                   True))
        self.validation_inputs = (self.compute_batches(tf.convert_to_tensor(self.validation_inputs),
                                                       validation_batch_nr,
                                                       validation_batch_size,
                                                       False))
        self.validation_targets = (self.compute_batches(tf.convert_to_tensor(self.validation_targets),
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
            X_test, y_test = (self.test_inputs, self.test_targets)

        self._coord = tf.train.Coordinator()
        self._threads_train = qr_train.create_threads(session, self._coord, start=True)
        self._threads_validation = qr_validation.create_threads(session, self._coord, start=True)

        return (X_train, y_train), (X_validation, y_validation), (X_test, y_test), (train_batch_nr, validation_batch_nr, 1)

    def free_threads(self):
        self._coord.request_stop()
        self._coord.join(self._threads_train)
        self._coord.join(self._threads_validation)

# # %% This part is used for debugging only
# with tf.Session() as ses:
#     ses.run(tf.global_variables_initializer())
#     rd = Reader(data_file=data_file, batch_size=20, validation_split=0.2, test_split=0.2, variance_threshold=0.95)
#     (train_in, train_out), (valid_in, valid_out), (test_in, test_out) , (_,_,_)= rd.batch_producer(ses)
#     print(ses.run(train_in).shape)
#     print(ses.run(train_out).shape)
#     print(ses.run(train_in)[0][0:10])
#     print(ses.run(test_out)[0])
#     print(ses.run(train_in)[0][0:10])
#     print(ses.run(test_out)[0])

#     print(ses.run(valid_in).shape)
#     print(ses.run(valid_out).shape)
#     print(ses.run(valid_in)[0][0:10])
#     print(ses.run(valid_out)[0])
#     print(ses.run(valid_in)[0][0:10])
#     print(ses.run(valid_out)[0])

#     print(ses.run(test_in).shape)
#     print(ses.run(test_out).shape)
#     print(ses.run(test_in)[0][0:10])
#     print(ses.run(test_out)[0])
#     print(ses.run(test_in)[0][0:10])
#     print(ses.run(test_out)[0])

#     rd.free_threads()


# #%%


#%%
