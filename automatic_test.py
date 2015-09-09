#!/usr/bin/python
#coding:utf-8
from __future__ import division
from abc import ABCMeta, abstractmethod
from extract_features import get_intersection_angle, get_curve_secant_line,\
    sample_line, _feature_types, get_matrices_from_samples, get_samples_from_matrices,\
    get_predicted_line, plot_intersection, _feature_types
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing
import random
import pickle

class PredictionAlgorithm(object):
    __metaclass__ = ABCMeta
    name = ''

    def train(self, samples):
        pass

    @abstractmethod
    def predict(self, sample):
        pass

    def get_name(self):
        return self.name

def normalize_features(samples):
    """Normalize all the feature vectors in samples"""
    X, y = get_matrices_from_samples(samples)
    X = sklearn.preprocessing.normalize(X, axis=1, copy=False)
    return get_samples_from_matrices(X, y, samples)

def get_partitioned_samples(samples, train_ratio):
    """Randomize the given samples and partition them in train and test
    samples using the train_ratio"""
    sample_count = len(samples)
    print 'Total number of samples:', sample_count
    train_sample_count = int(round(sample_count * train_ratio))
    indices = range(sample_count)
    random.shuffle(indices)
    train_indices = indices[:train_sample_count]
    test_indices = indices[train_sample_count:]
    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]
    return train_samples, test_samples

def train(algorithms, train_samples):
    for algo in algorithms:
        algo.train(train_samples)

def test(algorithms, test_samples):
    for algo in algorithms:
        cumulated_mse = 0.
        average_mse = 0.
        min_mse = None
        max_mse = None
        for test_sample in test_samples:
            y_true = test_sample['y']
            y_pred = algo.predict(test_sample)
            mse = mean_squared_error(y_true, y_pred)
            cumulated_mse += mse
            average_mse += mse/len(test_samples)
            if min_mse != None:
                min_mse = min(min_mse, mse)
            else:
                min_mse = mse
            if max_mse != None:
                max_mse = max(max_mse, mse)
            else:
                max_mse = mse
        print 'Test with algorithm:', algo.get_name()
        print 'Cumulated MSE:', cumulated_mse
        print 'Average MSE:', average_mse
        print 'Minimum MSE:', min_mse
        print 'Maximum MSE:', max_mse

def test_plot(algorithms, test_samples):
    for s in test_samples:
        predicted_lines = []
        for algo in algorithms:
            y_pred = algo.predict(s)
            predicted_lines.append(get_predicted_line(s['geometry']['curve_secant'], y_pred, s['X'][_feature_types.index('intersection_angle')]))
        plot_intersection(s['geometry']['entry_line'], s['geometry']['exit_line'],\
                        s['geometry']['curve_secant'], s['geometry']['track_line'], predicted_lines)

def load_samples(fn):
    with open(fn, 'r') as f:
        samples = pickle.load(f)
    return samples
