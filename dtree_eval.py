'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import tree
from sklearn.metrics import accuracy_score




def evaluatePerformance(numTrials=100):
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    numTrials = 10
    num_folds = 10
    total_trials = numTrials * num_folds

    total_accuracy_regular_tree = 0
    total_accuracy_stump = 0
    total_accuracy_DT3 = 0

    accuracy_regular_tree = []
    accuracy_stump = []
    accuracy_DT3 = []

    for i in range(0, numTrials):

        # shuffle the data
        idx = np.arange(n)
        np.random.seed()
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        fold_length = math.floor(len(X)/num_folds)
        test_index = 0

        

        for j in range(0, num_folds):

            Xtest = X[test_index:(test_index + fold_length), :]
            ytest = y[test_index:(test_index + fold_length), :]
            Xtrain_front = X[0:test_index, :]
            ytrain_front = y[0:test_index, :]

            test_index += fold_length

            Xtrain_end = X[(test_index + fold_length):, :]
            ytrain_end = y[(test_index + fold_length):, :]

            Xtrain = np.concatenate((Xtrain_front, Xtrain_end))
            ytrain = np.concatenate((ytrain_front, ytrain_end))


            # train the decision tree
            regular_tree = tree.DecisionTreeClassifier()
            regular_tree = regular_tree.fit(Xtrain,ytrain)

            # train the decision stump
            stump = tree.DecisionTreeClassifier(max_depth=1)
            stump = stump.fit(Xtrain, ytrain)

            # train the DT3
            dt3 = tree.DecisionTreeClassifier(max_depth=3)
            dt3 = dt3.fit(Xtrain, ytrain)

            # output predictions on the remaining data
            y_pred_regular_tree = regular_tree.predict(Xtest)
            y_pred_stump = stump.predict(Xtest)
            y_pred_DT3 = dt3.predict(Xtest)

            # compute the training accuracy of the model
            overall_test_index = (i * num_folds) + j
            accuracy_regular_tree.append(accuracy_score(ytest, y_pred_regular_tree))
            accuracy_stump.append(accuracy_score(ytest, y_pred_stump))
            accuracy_DT3.append(accuracy_score(ytest, y_pred_DT3))

            # total_accuracy_regular_tree += accuracy_score(ytest, y_pred_regular_tree)
            # total_accuracy_stump += accuracy_score(ytest, y_pred_stump)
            # total_accuracy_DT3 += accuracy_score(ytest, y_pred_DT3)
        
    
    # TODO: update these statistics based on the results of your experiment

    meanDecisionTreeAccuracy = np.sum(accuracy_regular_tree)/total_trials
    stddevDecisionTreeAccuracy = np.std(accuracy_regular_tree)
    meanDecisionStumpAccuracy = np.sum(accuracy_stump)/total_trials
    stddevDecisionStumpAccuracy = np.std(accuracy_stump)
    meanDT3Accuracy = np.sum(accuracy_DT3)/total_trials
    stddevDT3Accuracy = np.std(accuracy_DT3)

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")"
    print "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")"
    print "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")"
# ...to HERE.
