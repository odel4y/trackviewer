#!/usr/bin/python
#coding:utf-8
from __future__ import division
from abc import ABCMeta, abstractmethod
from extract_features import get_intersection_angle, get_curve_secant_line, sample_line, _feature_types
from sklearn.metrics import mean_squared_error

class PredictionAlgorithm(object):
    __metaclass__ = ABCMeta
    name = ''

    @abstractmethod
    def predict(self, samples):
        pass

    def train(self, samples):
        pass

    def get_name(self):
        return self.name

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
