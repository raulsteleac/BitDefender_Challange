#%% Change working directory from the workspace root to the ipynb file location.
import tensorflow as tf
import numpy as np
import os
try:
    os.chdir(os.path.join(
        os.getcwd(), 'challange'))
    print(os.getcwd())
except:
    pass

data_file_train = 'challenge_train.csv'
data_file_test  = 'challenge_test.csv'
from reader import Reader
tf.logging.set_verbosity(tf.logging.ERROR)

class BitDefender_Data_Producer(object):
      def __init__(self, config):
            self._data_producer = Reader(config.feature_extraction_method ,config.data_file, config.test_file, config.batch_size, config.validation_split, config.variance_threshold)

      def import_data(self, session):
            (self.train_inputs, self.train_targets), (self.validation_inputs, self.validation_targets), (self.test_inputs, self.test_targets), (self.train_batch_nr, self.validation_batch_nr, self.test_batch_nr) = self._data_producer.batch_producer(session)
      @property
      def train_data(self):
            return self.train_inputs, self.train_targets, self.train_batch_nr

      @property
      def validation_data(self):
            return self.validation_inputs, self.validation_targets, self.validation_batch_nr

      @property
      def test_data(self):
            return self.test_inputs, self.test_targets, self.test_batch_nr

      def close(self):
            self._data_producer.free_threads()

class BitDefender_Challanger(object):
      def __init__(self, inputs, targets=None, batch_nr=1, is_training=False, learning_rate=0, model_op_name=None):
            self._learning_rate = learning_rate
            self.init = tf.random_uniform_initializer(-0.1, 0.1)
            self.model_op_name = model_op_name

            self.inputs = inputs
            self.targets = targets
            self.batch_nr = batch_nr
            self.is_training = is_training        

      def initialize_variables(self, session):
            session.run(tf.global_variables_initializer())

      def model(self):
            """ CREATE CHALLANGE MODEL WITH 4 HIDDEN LAYERS FOR CLASSIGICATION USING SIGMOID CROSS ENTROPY LOSS
                cross_entropy = sigmoid cross entropy loss
                Y_ = predictions
                is_correct = tensor of 1s and 0s representing the correct and incorrect predictions
                Class data members:
                self.accuracy = actual calculated accuracy 
                self.optimizer = minimization tensor op using Adam Optimizer
            """
            with tf.variable_scope("BitDefender_Challanger", reuse=tf.AUTO_REUSE, initializer=self.init):
                  hidden_layer_1 = tf.layers.dense(self.inputs, 128, activation=tf.nn.relu, name="First_Fully_Connected_Layer")
                  hidden_layer_1 = self.batch_normalization(hidden_layer_1)
                  hidden_layer_2 = tf.layers.dense(hidden_layer_1, 96, activation=tf.nn.relu, name="Second_Fully_Connected_Layer")
                  hidden_layer_2 = self.batch_normalization(hidden_layer_2)
                  hidden_layer_3 = tf.layers.dense(hidden_layer_2, 64, activation=tf.nn.relu, name="Third_Fully_Connected_Layer" )
                  hidden_layer_3 = self.batch_normalization(hidden_layer_3)
                  hidden_layer_4 = tf.layers.dense(hidden_layer_3, 32, activation=tf.nn.relu, name="Forth_Fully_Connected_Layer" )
                  output_layer   = tf.layers.dense(hidden_layer_4,  1, name="Output_Layer")   

                  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=output_layer)
                  Y_ = tf.nn.sigmoid(output_layer)
                  is_correct = tf.equal(tf.round(Y_), self.targets)
                  self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

                  if not self.is_training:
                        return

                  tf.summary.scalar('Accuracy', self.accuracy)
                  adam_opt = tf.train.AdamOptimizer(self._learning_rate)
                  self.optimizer = adam_opt.minimize(cross_entropy)

      def batch_normalization(self, batch): 
            """ COMPUTE BATCH NORMALIZAION ON THE BATCH GIVEN AS INPUT TO EACH HIDDEN LAYER
                Return:
                Returns the transformed batch with mean 0 and stdev 1  
            """          
            means = tf.reduce_mean(batch, 0)
            stdev = tf.reduce_mean(np.power((batch - means), 2), 0) + 0.00001
            return (batch - means) / (tf.sqrt(stdev))

      @property
      def running_ops(self):
            """ RETURNS THE TENSORS THAT NEED TO BE COMPUTED. DEPENDENCIES WILL MAKE ALL THE OTHER NECESSARY TENSORS BE COMPUTED TOO.
            """
            if self.is_training:
                  return {
                        "accuracy": self.accuracy,
                        "optimizer": self.optimizer
                  }
            else:
                  return {
                        "inputs": self.inputs,
                        "accuracy": self.accuracy
                  }

      def run_model(self, session, writer, merged_summaries):
            """ RUNNING MODEL ON CURRENT CONFIGURATION 
                This method is computing training, validation or testing depending on what model is calling it.
            """
            print("\n %s just started !" % self.model_op_name)
            accuracy = 0.0
            for batch in range(self.batch_nr):
                  vals = session.run(self.running_ops)
                  accuracy += vals["accuracy"]
                  if (self.batch_nr // 10)  != 0 and batch % (self.batch_nr // 10) == 0:
                        print("-----------> Batch number : %d Current Accuracy : %f" % (batch / (self.batch_nr // 10),  vals["accuracy"]))
                  writer.add_summary(session.run(merged_summaries))
            print("############### %s Total Accuracy = %lf \n" % (self.model_op_name, (accuracy / self.batch_nr)))

      def debug_print(self, session):
            print(self.inputs.shape)
            print(self.targets.shape)
            print(session.run(self.inputs)[0][0:10])
            print(session.run(self.inputs)[0][0:10])        

class NormalConfig(object):
      feature_extraction_method = 'Autoencoder'
      data_file = data_file_train
      test_file = data_file_test
      batch_size = 200
      validation_split = 0.3
      variance_threshold = 0.96

#%%
def main():
      ses = tf.Session()
      # Compute train, validation and testing data
      btc_dp = BitDefender_Data_Producer(NormalConfig())
      btc_dp.import_data(ses)
      train_inputs, train_target, train_batch_nr = btc_dp.train_data
      validation_inputs, validation_targets, validation_batch_nr = btc_dp.validation_data
      test_inputs, test_targets, test_batch_nr = btc_dp.test_data

      # Create the three model objects with the needed inputs
      btc_train_model      = BitDefender_Challanger(train_inputs, train_target, train_batch_nr, True, 0.001, model_op_name="Training")
      btc_validation_model = BitDefender_Challanger(validation_inputs, validation_targets, validation_batch_nr, model_op_name="Validation")
      btc_test_model       = BitDefender_Challanger(test_inputs, test_targets, test_batch_nr, model_op_name="Testing")

      # Create model graphs for the 3 ops
      btc_train_model.model()
      btc_validation_model.model()
      btc_test_model.model()

      # Create writer and summaries for tensorboard
      writer = tf.summary.FileWriter('./graphs', ses.graph)
      merged_summaries = tf.summary.merge_all()

      # Initialize model variables and run the models. Only training model need initialization because the rest use the traind weights and biases.
      btc_train_model.initialize_variables(ses)
      epochs = 5
      for epoch in range(epochs):
            print("\n-----------> Epoch %d" % epoch)
            btc_train_model.run_model(ses,writer, merged_summaries)
            btc_validation_model.run_model(ses, writer, merged_summaries)
      writer = tf.summary.FileWriter('./graphs', ses.graph)
      btc_test_model.run_model(ses, writer, merged_summaries)

      btc_dp.close()
      ses.close()

if __name__ == "__main__":
    main()



#%%
