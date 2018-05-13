import os 
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data as cd


import tensorflow as tf

 

# hello = tf.constant('Hello tf!')
# sess = tf.Session()
# print(sess.run(hello))

data_folder = '/Users/dmitrykazakov/Desktop/Studium/MSc/2. Semester/ML/projects/task4_s8n2k3nd/data/ex4/'

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



def main():

	# import datasets

	train_labeled = pd.read_hdf(data_folder + "train_labeled.h5", "train")
	train_unlabeled = pd.read_hdf(data_folder + "train_unlabeled.h5", "train")
	test = pd.read_hdf(data_folder + "test.h5", "test")

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


	batch_size = 100


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
	
	prediction_unlabeled = classifier.predict(
		input_fn=lambda:eval_input_fn(unlabeled_x,
										batch_size=batch_size))

	predicted_unlabeled = list(prediction_unlabeled) 

	class_id = np.zeros(len(predicted_unlabeled),)
	for ii in range(0,len(predicted_unlabeled)-1):
		class_id[ii] = predicted_unlabeled[ii]['class_ids']
		
	print('\nDONE')

	
	# extended set

	feature_values_extended_train = np.concatenate((feature_values_train_set, feature_values_unlabeled_set), axis=0)
	labels_extended_train = np.concatenate((labels_train_set, class_id), axis=0)

	# convert again to Tensorflow dataset
	extended_train_x, extended_train_y = input_eval_set(feature_names, feature_values_extended_train, labels_extended_train)

	# train on extended dataset
	print('\nTraining on extended dataset...')
	extended_classifier = tf.estimator.DNNClassifier(
		feature_columns=feature_columns,
		hidden_units=[2048,1024,512],
		n_classes=10)

	extended_classifier.train(
		input_fn=lambda:train_input_fn(extended_train_x,extended_train_y,batch_size),
		steps=10000)
	print('\nDONE')


	# now predict on test data
	# prediction = classifier.predict(
	# 	input_fn=lambda:eval_input_fn(test_x,
	# 									batch_size=batch_size)) 

	# now predict on test data
	prediction = extended_classifier.predict(
		input_fn=lambda:eval_input_fn(test_x,
										batch_size=batch_size))


	predicted = list(prediction)

	# print(len(predicted))

	class_id = np.zeros(len(predicted),)
	for ii in range(0,len(predicted)-1):
		class_id[ii] = predicted[ii]['class_ids']
	print(class_id)

	yPred = pd.DataFrame(class_id, index=test.index, columns=['y'])
	yPred.index.name = 'Id'
	yPred.to_csv(data_folder + 'self-training_v1.csv')

	



	# print(labels_train_set[10])



	return






if __name__ == '__main__':
	main()
	print('\nDone!')