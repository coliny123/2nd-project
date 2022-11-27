#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/coliny123/2nd-project.git

import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


def load_dataset(dataset_path):
	#To-Do: Implement this function\
        dataset_df = pd.DataFrame(dataset_path) 

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
        n_feats = dataset_df.data.shape[1]
        n_class0 = count(dataset_df.target[0])
        n_class0 = count(dataset_df.target[1])
        
        
        

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
        x_train, x_test, y_train, y_test =  train_test_split(dataset_df.data, dataset_df.target, testset_size = sys.argv[2], random_state = 2)
        


def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
        StandardScaler().fit(x_train).transform(x_train)
        pipe = make_pipeline(
                StandardScaler(),
                DecisionTreeClassifier()
        )
        pipe.fit(x_train, y_train)
        

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
        StandardScaler().fit(x_train).transform(x_train)
        pipe = make_pipeline(
                StandardScaler(),
                RandomForestClassifer()
        )
        pipe.fit(x_train, y_train)
        

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
        StandardScaler().fit(x_train).transform(x_train)
        pipe = make_pipeline(
                StandardScaler(),
                SVC()
        )
        pipe.fit(x_train, y_train)
        
        

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)
	
	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
