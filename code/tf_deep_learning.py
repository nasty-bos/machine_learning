import os 
import numpy as np
import pandas as pd
import data as cd
import data as dt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import logging 

import tensorflow as tf

 

# hello = tf.constant('Hello tf!')
# sess = tf.Session()
# print(sess.run(hello))

log = logging.getLogger(__name__)

#data_folder = '/Users/dmitrykazakov/Desktop/Studium/MSc/2. Semester/ML/projects/task4_s8n2k3nd/data/ex4/'
data_folder = os.path.join(dt.data_dir(), dt.DataSets.EX4.value)

def input_eval_set(feature_names, feature_values, class_labels=None):
	features = dict(zip(feature_names, np.matrix(feature_values).transpose().tolist()))
	if class_labels is None: 
		return features
	else:
		labels = np.int32(np.array(class_labels))
		return features, labels



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

def eval_input_fn(features, batch_size):
    
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(features)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset



def naive_bayes_class():

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
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()

	scaler.fit(train_labeled.loc[:, feature_names])
	feature_values_train_set = scaler.transform(feature_values_train_set)
	feature_values_unlabeled_set = scaler.transform(feature_values_unlabeled_set)
	feature_values_test_set = scaler.transform(feature_values_test_set)

	gnb = GaussianNB()

	class_id = gnb.fit(feature_values_train_set, labels_train_set).predict(feature_values_test_set)

	yPred = pd.DataFrame(class_id, index=test.index, columns=['y'])
	yPred.index.name = 'Id'
	yPred.to_csv(os.path.join(data_folder, 'NB.csv'))


def insample_learn():
	# import datasets

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
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()

	scaler.fit(train_labeled.loc[:, feature_names])
	feature_values_train_set = scaler.transform(feature_values_train_set)
	feature_values_unlabeled_set = scaler.transform(feature_values_unlabeled_set)
	feature_values_test_set = scaler.transform(feature_values_test_set)

	##
	log.info('train-test-split')
	X_train, X_test, y_train, y_test = train_test_split(
		feature_values_train_set, 
		labels_train_set.values.flatten(), 
		test_size=0.20, 
		random_state=12357, 
		stratify=labels_train_set.values.flatten()
	)

	# convert to TensorFlow datasets
	train_x, train_y = input_eval_set(feature_names, X_train, y_train)
	oos_x, oos_y = input_eval_set(feature_names, X_test, y_test)
	unlabeled_x = input_eval_set(feature_names, feature_values_unlabeled_set) 
	test_x = input_eval_set(feature_names, feature_values_test_set)
	

	feature_columns = []
	for key in train_x:
		feature_columns.append(tf.feature_column.numeric_column(key=key))


	batch_size = 10


	classifier = tf.estimator.DNNClassifier(
		feature_columns=feature_columns,
		hidden_units=[2048,1024,512],
		n_classes=10)

	# use self-training: first train on labeled data, then categorize unlabeled data and train again on a combined dataset

	print('\nTraining on labeled data...')
	classifier.train(
		input_fn=lambda:train_input_fn(train_x,train_y,batch_size),
		steps=10000)
	print('\nDONE \nClassifying ulabeled data based on the derived model...')
	
	prediction_labeled = classifier.predict(
		input_fn=lambda:eval_input_fn(oos_x,
										batch_size=batch_size))

	predicted_labeled = list(prediction_labeled)

	class_id = np.zeros(len(predicted_labeled),)
	for ii in range(0,len(predicted_labeled)-1):
		class_id[ii] = predicted_labeled[ii]['class_ids']

	conf = confusion_matrix(oos_y, np.int32(np.array(class_id)),labels=np.arange(0,10))

	write_matrix = pd.DataFrame(conf)
	write_matrix.to_csv(os.path.join(data_folder, 'confusion_matrix.csv'))	

	log.debug("MSE: %.5f", mean_squared_error(train_y, np.int32(np.array(class_id))))





def main():

	# import datasets

	train_labeled = pd.read_hdf(os.path.join(data_folder, "train_labeled.h5"), "train")
	train_unlabeled = pd.read_hdf(os.path.join(data_folder, "train_unlabeled.h5"), "train")
	test = pd.read_hdf(os.path.join(os.path.join(data_folder, "test.h5")), "test")

	# === shuffle data prior to fitting
	RM = np.random.RandomState(12357)
	train_labeled.index = RM.permutation(train_labeled.index)
	
	RM = np.random.RandomState(12357)
	train_unlabeled.index = RM.permutation(train_unlabeled.index)


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
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()

	scaler.fit(train_labeled.loc[:, feature_names])
	feature_values_train_set = scaler.transform(feature_values_train_set)
	feature_values_unlabeled_set = scaler.transform(feature_values_unlabeled_set)
	feature_values_test_set = scaler.transform(feature_values_test_set)

	# # perform clustering on unlabeled data
	# from sklearn.cluster import KMeans
	# kmeans = KMeans(n_clusters = 10, random_state = 0).fit(train_x_unlabeled)


	# PCA or LDA on labeled data
	# from sklearn.decomposition import PCA
	# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	
	# pca = PCA(n_components = 5, svd_solver = 'auto')
	# feature_values_train_set = pca.fit_transform(feature_values_train_set)
	# lda = LinearDiscriminantAnalysis(n_components = 5)

	# print(feature_values_train_set[1,:])




	# convert to TensorFlow datasets
	train_x, train_y = input_eval_set(feature_names, feature_values_train_set, labels_train_set)
	
	unlabeled_x = input_eval_set(feature_names, feature_values_unlabeled_set) 
	test_x = input_eval_set(feature_names, feature_values_test_set)
	

	feature_columns = []
	for key in train_x:
		feature_columns.append(tf.feature_column.numeric_column(key=key))


	batch_size = 50


	classifier = tf.estimator.DNNClassifier(
		feature_columns=feature_columns,
		hidden_units=[2048,1024,512],
		n_classes=10)

	# use self-training: first train on labeled data, then categorize unlabeled data and train again on a combined dataset

	print('\nTraining on labeled data...')
	classifier.train(
		input_fn=lambda:train_input_fn(train_x,train_y,batch_size),
		steps=10000)
	print('\nDONE \nClassifying ulabeled data based on the derived model...')
	
	# prediction_unlabeled = classifier.predict(
	# 	input_fn=lambda:eval_input_fn(unlabeled_x,
	# 									batch_size=batch_size))

	# predicted_unlabeled = list(prediction_unlabeled) 

	# class_id = np.zeros(len(predicted_unlabeled),)
	# for ii in range(0,len(predicted_unlabeled)-1):
	# 	class_id[ii] = predicted_unlabeled[ii]['class_ids']
		
	# print('\nDONE')

	# # drop samples classified to 9
	# to_keep = [i for i, label in enumerate(class_id) if label != 9]
	# feature_values_unlabeled_set = feature_values_unlabeled_set[to_keep, :]
	# class_id = class_id[to_keep]


	
	# extended set

	# feature_values_extended_train = np.concatenate((feature_values_train_set, feature_values_unlabeled_set), axis=0)
	# labels_extended_train = np.concatenate((labels_train_set, class_id), axis=0)

	# # convert again to Tensorflow dataset
	# extended_train_x, extended_train_y = input_eval_set(feature_names, feature_values_extended_train, labels_extended_train)

	# # train on extended dataset
	# print('\nTraining on extended dataset...')
	# extended_classifier = tf.estimator.DNNClassifier(
	# 	feature_columns=feature_columns,
	# 	hidden_units=[2048,1024,512],
	# 	n_classes=10)

	# extended_classifier.train(
	# 	input_fn=lambda:train_input_fn(extended_train_x,extended_train_y,batch_size),
	# 	steps=10000)
	# print('\nDONE')


	# now predict on test data
	prediction = classifier.predict(
		input_fn=lambda:eval_input_fn(test_x,
										batch_size=batch_size)) 

	# now predict on test data
	# prediction = extended_classifier.predict(
		# input_fn=lambda:eval_input_fn(test_x,
		# 								batch_size=batch_size))


	predicted = list(prediction)

	# print(len(predicted))

	class_id = np.zeros(len(predicted),)
	for ii in range(0,len(predicted)-1):
		class_id[ii] = predicted[ii]['class_ids']
	print(class_id)

	yPred = pd.DataFrame(class_id, index=test.index, columns=['y'])
	yPred.index.name = 'Id'
	yPred.to_csv(os.path.join(data_folder, 'Batch_size_50.csv'))

	

	return


if __name__ == '__main__':
	# naive_bayes_class()
	main()
	# insample_learn()
	print('\nDone!')
