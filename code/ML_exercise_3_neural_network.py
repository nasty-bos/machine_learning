import data as cd 

import tensorflow as tf 
import pandas 
import numpy
import os


#######################################################################
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


#######################################################################
def main():

	# === Data
	train = pandas.read_hdf(os.path.join(cd.data_dir(), cd.DataSets.EX3.value, 'train.h5'), key='train')
	test = pandas.read_hdf(os.path.join(cd.data_dir(), cd.DataSets.EX3.value, 'test.h5'), key='test')

	train_tensor = tf.convert_to_tensor(train.as_matrix())
	batches = tf.split(train, num_or_size_splits=36, axis=0)

	print(batches)

	features = ["x%i" %i for i in numpy.arange(1,101,1)]
	target = ['y']

	n, F = train[features].shape
	C = train[target].unique

	# ==== Parameters
	learning_rate = 0.1
	num_steps = 500
	batch_size = 128
	display_step = 100

	# === Network Parameters
	n_hidden_1 = 256 # 1st layer number of neurons
	n_hidden_2 = 256 # 2nd layer number of neurons
	num_input = F
	num_classes = len(C) # MNIST total classes (0-9 digits)

	# === tf Graph input - reserve space / decalre 
	X = tf.placeholder("float", [None, num_input])
	Y = tf.placeholder("float", [None, num_classes])	

	# === Store layers weight & bias
	weights = {
	    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
	    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
	}
	biases = {
	    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	    'out': tf.Variable(tf.random_normal([num_classes]))
	}


	# === Construct model
	logits = neural_net(X)
	prediction = tf.nn.softmax(logits)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(
			logits=logits, 
			labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Start training
	with tf.Session() as sess:

	    # Run the initializer
	    sess.run(init)

	    for step in range(1, num_steps+1):
	    	for batch in batches:
		        batch_x, batch_y = mnist.train.next_batch(batch_size)
		        # Run optimization op (backprop)
		        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
		        if step % display_step == 0 or step == 1:
		            # Calculate batch loss and accuracy
		            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
		                                                                 Y: batch_y})
		            print("Step " + str(step) + ", Minibatch Loss= " + \
		                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
		                  "{:.3f}".format(acc))

	    print("Optimization Finished!")

	    # Calculate accuracy for MNIST test images
	    print("Testing Accuracy:", \
	        sess.run(accuracy, feed_dict={X: mnist.test.images,
	                                      Y: mnist.test.labels}))
	
	return {}

#######################################################################
if __name__ == "__main__":
	main()
	print('Done!')