from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import data as cd 
import pandas as pd 
import numpy as np
import os 

#######################################################################
def main():

	# Load the datasets
	trainSet = pd.read_csv(os.path.join(cd.data_dir(), cd.DataSets.EX2.value, 'train.csv'), header=0, index_col=0, float_precision='round_trip')
	testSet = pd.read_csv(os.path.join(cd.data_dir(), cd.DataSets.EX2.value, 'test.csv'), header=0, index_col=0, float_precision='round_trip')
	sampleSet = pd.read_csv(os.path.join(cd.data_dir(), cd.DataSets.EX2.value, 'sample.csv'), header=0, index_col=0, float_precision='round_trip')

	xColumns = ['x' + str(i+1) for i in range(16)]
	yColumns = ['y']

	# Now apply the transformations to the data:
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	
	scaler.fit(trainSet.loc[:, xColumns])
	scaledTrainSet = scaler.transform(trainSet.loc[:, xColumns])
	scaledTestSet = scaler.transform(testSet.loc[:, xColumns])

	# set up SVM classifier 
	svm = SVC()

	paramters = {
		'C': np.arange(1.85, 1.89, 0.001), 
		'kernel': ['rbf', 'poly']
	}
	optSVM = GridSearchCV(svm, paramters)
	optSVM.fit(scaledTrainSet, trainSet.loc[:, yColumns].as_matrix().flatten())

	#  MLP TRAINING
	optSVM.fit(scaledTrainSet, trainSet.loc[:, yColumns].as_matrix().flatten())

	# prediction in-sample
	yOpt_in_sample = optSVM.predict(scaledTrainSet)

	# classification statistics
	dim = len(yOpt_in_sample)
	inSampleConfusionMatrix = confusion_matrix(trainSet.loc[:, yColumns], yOpt_in_sample)
	accuracy = np.sum(np.diag(inSampleConfusionMatrix)) / dim
	print("Using optimal parameter alpha, model accuracy %.4f " %accuracy)

	# prediction
	y_pred = optSVM.predict(scaledTestSet)

	# write to pandas Series object 
	yPred = pd.DataFrame(y_pred, index=testSet.index, columns=['y'])
	yPred.to_csv(os.path.join(cd.data_dir(), cd.DataSets.EX2.value, 'svm_classifier_python.csv'))

	# print out results of GridSearchCV
	print(pd.DataFrame.from_dict(optSVM.cv_results_))


#######################################################################
if __name__ == "__main__":
	main()