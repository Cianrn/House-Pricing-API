import tensorflow as tf
import time
import numpy as np

class NeuralNet:
	def __init__(self, learning_rate):
		self.save_epoch = 10
		self.save_path = './Boston_models/NeuralNet/'

		self.x_inputs = tf.placeholder(tf.float32, [None, 13])
		self.y_output = tf.placeholder(tf.float32, [None, 1])
		self.sess = tf.InteractiveSession()

		self.output = self.build_nn()

		loss = tf.square(self.output - self.y_output)
		self.error = tf.reduce_mean(loss)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		self.train_op = optimizer.minimize(self.error)

		self.sess.run(tf.global_variables_initializer())

	def build_nn(self):
		layer1 = tf.layers.dense(self.x_inputs, units=24, activation=tf.nn.relu)
		layer2 = tf.layers.dense(layer1, units=12, activation=tf.nn.relu)
		layer3 = tf.layers.dense(layer2, units=6, activation=tf.nn.relu)
		return tf.layers.dense(layer3, units=1)

	def train(self, inputs, labels, x_val, y_val, epochs, batch_size):
		epoch_error = []
		val_error = []
		for epoch in range(epochs):
			start_time = time.time()
			total_loss = 0
			batch_num = 0
			for batch in range(0, len(inputs), batch_size):
				x = inputs[batch:batch+batch_size]
				y = labels[batch:batch+batch_size]
				_, err = self.sess.run([self.train_op, self.error], feed_dict = {self.x_inputs: x,
																				self.y_output: y})
				total_loss += err
				batch_num += 1

			verr = self.sess.run([self.error], feed_dict = {self.x_inputs: x_val,
																	self.y_output: y_val})

			epoch_error.append(total_loss/batch_num)
			val_error.append(verr)

			print("Epoch {0} Train Loss: {1:1f} Validation Loss: {2} Time: {3}".format(epoch+1, total_loss/batch_num, verr[0], time.time()-start_time))
			if epoch % self.save_epoch == 0:
				self.save_model()

		writer = tf.summary.FileWriter('./boston_models/tb_nn', self.sess.graph) # Tensorboard
	
		return epoch_error, val_error

	def save_model(self):
		saver = tf.train.Saver()
		saver.save(self.sess, self.save_path + 'model_nn.ckpt')

	def load_model(self):
		saver = tf.train.Saver()
		saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))

	def predict(self, inputs):
		inputs = np.reshape(inputs, [1, 13])
		prediction = self.sess.run([self.output], feed_dict={self.x_inputs: inputs}) 
		return prediction


class LinearReg:
	def __init__(self, learning_rate):
		self.save_epoch = 10

		self.x_inputs = tf.placeholder(tf.float32, [None, 13])
		self.y_output = tf.placeholder(tf.float32, [None, 1])
		self.sess = tf.InteractiveSession()

		self.weights = tf.Variable(tf.truncated_normal([13, 1]))
		self.biases = tf.Variable(tf.zeros([13]))

		self.output = self.build_lm()
		self.save_path = './Boston_models/LinearReg/'

		loss = tf.square(self.output - self.y_output)
		self.error = tf.reduce_mean(loss)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.train_op = optimizer.minimize(self.error)

		self.sess.run(tf.global_variables_initializer())

	def build_lm(self):
		return tf.add(tf.matmul(self.x_inputs, self.weights), self.biases)
		# return tf.layers.dense(self.x_inputs, units=1)

	def train(self, inputs, labels, x_val, y_val, epochs, batch_size):
		epoch_error = []
		val_error = []
		for epoch in range(epochs):
			start_time = time.time()
			total_loss = 0
			batch_num = 0
			for batch in range(0, len(inputs), batch_size):
				x = inputs[batch:batch+batch_size]
				y = labels[batch:batch+batch_size]
				_, err = self.sess.run([self.train_op, self.error], feed_dict = {self.x_inputs: x,
																				self.y_output: y})
				total_loss += err
				batch_num += 1

			verr = self.sess.run([self.error], feed_dict = {self.x_inputs: x_val,
																	self.y_output: y_val})

			epoch_error.append(total_loss/batch_num)
			val_error.append(verr)

			print("Epoch {0} Train Loss: {1:1f} Validation Loss: {2} Time: {3}".format(epoch+1, total_loss/batch_num, verr[0], time.time()-start_time))
			if epoch % self.save_epoch == 0:
				self.save_model()

		writer = tf.summary.FileWriter('./boston_models/tb_lm', self.sess.graph) # Tensorboard
		self.sess.close()
		return epoch_error, val_error

	def save_model(self):
		saver = tf.train.Saver()
		saver.save(self.sess, self.save_path + 'model_lm.ckpt')



	def load_model(self):
		saver = tf.train.Saver()
		saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))

	def predict(self, inputs):

		inputs = np.reshape(inputs, [1, 13])
		prediction = self.sess.run([self.output], feed_dict={self.x_inputs: inputs}) 
		return prediction