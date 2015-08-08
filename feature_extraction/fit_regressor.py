#!/usr/bin/python
#coding:utf-8
import pickle
import sys
import argparse
import random
import numpy as np
import sklearn.preprocessing
import sklearn.ensemble
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

parser = argparse.ArgumentParser(description='No description')
parser.add_argument('-d', '--database', default="../data/training_data/samples.pickle")
parser.add_argument('-r', '--ratio', default=0.9, type=float, help="Ratio of training samples/total samples. Rest is for testing")
parser.add_argument('-m', '--model', type=str)
parser.add_argument('--plot', action='store_true', help="Plot the results during testing")

def get_randomized_samples(X, y, sample_ratio):
    """Return both a training and a test set in randomized order from the samples"""
    sample_count = len(X)
    print 'Total number of samples:', sample_count
    train_sample_count = int(round(float(sample_count) * sample_ratio))
    indices = range(sample_count)
    random.shuffle(indices)
    train_indices = indices[:train_sample_count]
    test_indices = indices[train_sample_count:]
    X_train = np.zeros((train_sample_count, X.shape[1]))
    y_train = np.zeros((train_sample_count, y.shape[1]))
    X_test = np.zeros((len(test_indices), X.shape[1]))
    y_test = np.zeros((len(test_indices), y.shape[1]))
    for i_dest, i_src in enumerate(train_indices):
        X_train[i_dest] = X[i_src]
        y_train[i_dest] = y[i_src]
    for i_dest, i_src in enumerate(test_indices):
        X_test[i_dest] = X[i_src]
        y_test[i_dest] = y[i_src]
    return X_train, y_train, X_test, y_test

def test_regressor(reg, X_test, y_test, plot):
    test_sample_count = len(X_test)
    print 'Testing regressor with %d samples...' % (test_sample_count)
    for i in xrange(test_sample_count):
        y_pred = reg.predict(X_test[i])
        y_true = y_test[i]
        print 'MSE:', mean_squared_error(y_true, y_pred[0])
        if plot:
            fig = pyplot.figure()
            pyplot.hold(True)
            pyplot.plot(range(len(y_true)), y_true, 'b.-')
            pyplot.plot(range(len(y_true)), y_pred[0], 'r.-')
            pyplot.legend(['Measurement', 'Prediction'])
            pyplot.show()

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.database, 'rb') as f:
        print 'Opening', args.database
        X,y = pickle.load(f)
    print 'Randomizing samples...'
    X_train, y_train, X_test, y_test = get_randomized_samples(X, y, args.ratio)
    print 'Normalizing features...'
    X_train = sklearn.preprocessing.normalize(X_train, axis=1, copy=False)
    X_test = sklearn.preprocessing.normalize(X_test, axis=1, copy=False)
    print 'Training regressor with %d samples...' % (len(X_train))
    reg = sklearn.ensemble.RandomForestRegressor()
    reg.fit(X_train, y_train)
    test_regressor(reg, X_test, y_test, plot=args.plot)
