#!/usr/bin/python
#coding:utf-8
from __future__ import division
from abc import ABCMeta, abstractmethod
from extract_features import get_intersection_angle, get_curve_secant_line, sample_line
from sklearn.metrics import mean_squared_error

class PredictionAlgorithm(object):
    __metaclass__ = ABCMeta
    name = ''

    @abstractmethod
    def predict(self, test_sample):
        pass

    def get_name(self):
        return self.name

def test(algorithms, test_samples):
    for this_algorithm in algorithms:
        cumulated_mse = 0.
        average_mse = 0.
        min_mse = None
        max_mse = None
        for test_samole in test_samples:
            entry_line = test_sample['geometry']['entry_line']
            exit_line = test_sample['geometry']['exit_line']
            track_line = test_sample['geometry']['track_line']
            intersection_angle = get_intersection_angle(entry_line, exit_line)
            curve_secant = get_curve_secant_line(entry_line, exit_line)
            y_true = sample_line(curve_secant, track_line, intersection_angle)
            predicted_line = this_algorithm.predict(test_sample)
            _, y_pred = sample_line(curve_secant, predicted_line, intersection_angle)
            mse = mean_squared_error(y_true, y_pred[0])
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
        print 'Test with algorithm:', this_algorithm.get_name()
        print 'Cumulated MSE:', cumulated_mse
        print 'Average MSE:', average_mse
        print 'Minimum MSE:', min_mse
        print 'Maximum MSE:', max_mse
