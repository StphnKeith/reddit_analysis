from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from scipy import stats
import numpy as np
import argparse
import sys
import os
import csv

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a
    floating point value
    '''
    dim = len(C)
    correct = 0
    total = 0

    for i in range(0, dim):
        correct += C[i, i]
        for j in range(0, dim):
            total += C[i, j]

    if total == 0:
        return 1.0
    return correct / total

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of
    floating point values
    '''
    recalls = []
    dim = len(C)

    for k in range(0, dim):
        correct = C[k, k]
        total = 0
        for j in range(0, dim):
            total += C[k, j]

        if total == 0:
            recalls.append(1.0)
        else:
            recalls.append(correct / total)

    return recalls

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list
    of floating point values
    '''
    precisions = []
    dim = len(C)

    for k in range(0, dim):
        correct = C[k, k]
        total = 0
        for j in range(0, dim):
            total += C[j, k]

        if total == 0:
            precisions.append(1.0)
        else:
            precisions.append(correct / total)

    return precisions

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''

    data = np.load(filename)['arr_0']
    X = data[:, 0:173]
    y = data[:, 173]

    np.nan_to_num(X, copy=False)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2 )

    lsvc = LinearSVC( loss='hinge' )
    rsvc = SVC( gamma=2.0, max_iter=1000 )
    rfc  = RandomForestClassifier( n_estimators=10, max_depth=5 )
    mlpc = MLPClassifier( alpha=0.05 )
    adbc = AdaBoostClassifier()

    lines = []
    clfr_num = 1

    classifiers = [lsvc, rsvc, rfc, mlpc, adbc]
    for clfr in classifiers:
        
        clfr.fit(X_train, y_train)
        y_pred = clfr.predict(X_test)
        C = confusion_matrix(y_test, y_pred)

        a = accuracy(C)
        r = recall(C)
        p = precision(C)

        line = [clfr_num, a]

        for k in r:
            line.append(k)

        for k in p:
            line.append(k)

        for row in C:
            for val in row:
                line.append(val)

        lines.append(line)

        clfr_num += 1

    iBest = 0
    max_a = 0
    for line in lines:
        if line[1] > max_a:
            iBest = line[0]

    with open("a1_3.1.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lines)

    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    clfs = {}
    clfs[1] = LinearSVC( loss='hinge' )
    clfs[2] = SVC( gamma=2.0, max_iter=1000 )
    clfs[3] = RandomForestClassifier( n_estimators=10, max_depth=5 )
    clfs[4] = MLPClassifier( alpha=0.05 )
    clfs[5] = AdaBoostClassifier()

    clfr = clfs[iBest]

    line = []
    increments = [1000, 5000, 10000, 15000, 20000]
    for i in increments:
        X_sample = X_train[0:i, :]
        y_sample = y_train[0:i]

        if i == 1000:
            X_1k = X_sample
            y_1k = y_sample

        
        clfr.fit(X_sample, y_sample)
        y_pred = clfr.predict(X_test)
        C = confusion_matrix(y_test, y_pred)
        a = accuracy(C)

        line.append(a)

    with open("a1_3.2.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerows([line])

    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    clfs = {}
    clfs[1] = LinearSVC( loss='hinge' )
    clfs[2] = SVC( gamma=2.0, max_iter=1000 )
    clfs[3] = RandomForestClassifier( n_estimators=10, max_depth=5 )
    clfs[4] = MLPClassifier( alpha=0.05 )
    clfs[5] = AdaBoostClassifier()

    clfr  = clfs[i]

    lines = []

    feat_counts = [5, 10, 20, 30, 40, 50]
    for i in range(0,2):
        if i == 0:
            tset = (X_1k, y_1k)
        else:
            tset = (X_train, y_train)

        for k in feat_counts:
            selector = SelectKBest(f_classif, k)

            if i == 0 and k == 5:
                X_new_1k = selector.fit_transform(tset[0], tset[1])
                X_new_test_1k = X_test[:, selector.get_support(indices=True)]
            elif i == 1 and k == 5:
                X_new_32k = selector.fit_transform(tset[0], tset[1])
                X_new_test_32k = X_test[:, selector.get_support(indices=True)]
            else:
                selector.fit_transform(tset[0], tset[1])

            pvals = selector.pvalues_

            if i == 1:
                line = []
                line.append(k)
                line += list(pvals[selector.get_support(indices=True)])
                lines.append(line)

    line = []
    for i in range(0,2):
        if i == 0:
            X = X_new_1k
            y = y_1k
            X_t = X_new_test_1k
        else:
            X = X_new_32k
            y = y_train
            X_t = X_new_test_32k

        clfr.fit(X, y)
        y_pred = clfr.predict(X_t)
        C = confusion_matrix(y_test, y_pred)
        a = accuracy(C)

        line.append(a)

    lines.append(line)

    with open("a1_3.3.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lines)


def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''

    data = np.load(filename)['arr_0']
    X = data[:, 0:173]
    y = data[:, 173]

    np.nan_to_num(X, copy=False)

    kf = KFold(n_splits=5, shuffle=True)

    folds = []
    for train_index, test_index in kf.split(X):
        folds.append( (train_index, test_index) )

    lsvc = LinearSVC( loss='hinge' )
    rsvc = SVC( gamma=2.0, max_iter=1000 )
    rfc  = RandomForestClassifier( n_estimators=10, max_depth=5 )
    mlpc = MLPClassifier( alpha=0.05 )
    adbc = AdaBoostClassifier()

    lines = []

    classifiers = [lsvc, rsvc, rfc, mlpc, adbc]
    for fold in folds:
        clfr_num = 1
        line = []
        train = fold[0]
        test = fold[1]
        for clfr in classifiers:
            clfr.fit( X[train], y[train] )
            y_pred = clfr.predict( X[test] )
            C = confusion_matrix( y[test], y_pred )
            a = accuracy(C)

            line.append(a)
            clfr_num += 1

        lines.append(line)

    line = []

    accuracy_array = np.array(lines)
    best_clfr_vec = accuracy_array[:, i-1]
    accuracy_array = np.delete(accuracy_array, i-1, 1)
    for j in range(0,4):
        vec = accuracy_array[:, j]
        S = stats.ttest_rel(best_clfr_vec, vec)
        line.append(S[1])

    lines.append(line)

    with open("a1_3.4.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lines)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # complete each classification experiment, in sequence.
    filename = args.input
    # Experiment 1
    (X_train, X_test, y_train, y_test, iBest) = class31(filename)

    # Experiment 2
    (X_1k, y_1k) = class32(X_train, X_test, y_train, y_test, iBest)

    # Experiment 3
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)

    # Experiment 4
    class34(filename, iBest)
