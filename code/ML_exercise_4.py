import os 
import numpy as np
import pandas as pd
import data as cd
import data as dt
import logging 
import tensorflow as tf
import time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split


#######################################################################
def input_eval_set(feature_names, feature_values, class_labels=None):
	features = dict(zip(feature_names, np.matrix(feature_values).transpose().tolist()))
	if class_labels is None: 
		return features
	else:
		labels = np.int32(np.array(class_labels))
		return features, labels



#######################################################################
def train_input_fn(features, labels, batch_size):

	# convert input to TensorFlow Dataset
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))

	# Shuffle, repeat and batch 
	return dataset.shuffle(1000).repeat().batch(batch_size)


# def eval_input_fn(features, labels, batch_size):
#     """An input function for evaluation or prediction"""
#     features=dict(features)
#     if labels is None:
#         # No labels, use only features.
#         inputs = features
#     else:
#         inputs = (features, labels)

#     # Convert the inputs to a Dataset.
#     dataset = tf.data.Dataset.from_tensor_slices(inputs)

#     # Batch the examples
#     assert batch_size is not None, "batch_size must not be None"
#     dataset = dataset.batch(batch_size)

#     # Return the dataset.
#     return dataset

#######################################################################
def eval_input_fn(features, batch_size):
    
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(features)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset



#######################################################################
def in_sample_tester():

	logger = logging.getLogger(__name__)

	## 
	logger.info('import datasets')
	data_folder = os.path.join(dt.data_dir(), dt.DataSets.EX4.value)
	train_labeled = pd.read_hdf(os.path.join(data_folder, "train_labeled.h5"), "train")
	train_unlabeled = pd.read_hdf(os.path.join(data_folder, "train_unlabeled.h5"), "train")
	test = pd.read_hdf(os.path.join(os.path.join(data_folder, "test.h5")), "test")

	# === shuffle data prior to fitting
	RM = np.random.RandomState(12357)
	train_labeled.index = RM.permutation(train_labeled.index)
	
	# extract column names, class labels
	feature_names = train_labeled.columns.values[1:]

	# labeled train set
	feature_values_train_set = train_labeled.loc[:, feature_names]
	labels_train_set = train_labeled.loc[:,train_labeled.columns.values[0]]

	# unlabeled train set
	feature_values_unlabeled_set = train_unlabeled.loc[:, feature_names]


	# test set
	feature_values_test_set = test.loc[:, feature_names]

	# Now apply the transformations to the data:
	scaler = StandardScaler()
	scaler.fit(train_labeled.loc[:, feature_names])
	feature_values_train_set = scaler.transform(feature_values_train_set)
	feature_values_unlabeled_set = scaler.transform(feature_values_unlabeled_set)
	feature_values_test_set = scaler.transform(feature_values_test_set)

	# === Stratefied sampling of labeled data-set
	logger.info('train-test-split')
	sss = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=RM)
	sss.get_n_splits(X, y)

	count=0
	for train_index, test_index in sss.split(X, y):

		count+=1
		logger.info('fold := %i' %count)
		start = time.time()

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]	
		
		# convert to TensorFlow datasets
		train_x, train_y = input_eval_set(feature_names, X_train, y_train)
		oos_x, oos_y = input_eval_set(feature_names, X_test, y_test)

		feature_columns = []
		for key in train_x:
			feature_columns.append(tf.feature_column.numeric_column(key=key))

		batch_size = 100

		classifier = tf.estimator.DNNClassifier(
			feature_columns=feature_columns,
			hidden_units=[2048,1024,512],
			n_classes=10)

		# use self-training: first train on labeled data, then categorize unlabeled data and train again on a combined dataset

		logger.info('\nTraining on labeled data...')

		classifier.train(
			input_fn=lambda:train_input_fn(train_x,train_y,batch_size),
			steps=10000)
		logger.info('\nDONE \nClassifying ulabeled data based on the derived model...')

		prediction_labeled = classifier.predict(
			input_fn=lambda:eval_input_fn(oos_x,
											batch_size=batch_size))

		predicted_labeled = list(prediction_labeled)

		class_id = np.zeros(len(predicted_labeled),)
		for ii in range(0,len(predicted_labeled)-1):
			class_id[ii] = predicted_labeled[ii]['class_ids']

		conf = confusion_matrix(oos_y, np.int32(np.array(class_id)),labels=np.arange(0,10))

		write_matrix = pd.DataFrame(conf)
		write_matrix.to_csv(os.path.join(data_folder, 'confusion_matrix_%i.csv') %count)	

		logger.info("MSE: %.5f" %(mean_squared_error(oos_y, np.int32(np.array(class_id)))))


#######################################################################
if __name__ == "__main__":

	logger = logging.getLogger('myapp')
	hdlr = logging.FileHandler('/var/tmp/myapp.log')
	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr) 
	logger.setLevel(logging.WARNING)	


	print('complete!')