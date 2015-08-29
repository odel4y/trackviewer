#!/usr/bin/python
#coding:utf-8
from abc import ABCMeta, abstractmethod
from extract_features import get_intersection_angle, get_curve_secant_line, sample_line

_test_sample = {
    'geometry': {
        'entry_line': None,
        'exit_line': None
    },
    'tags': {
        'maxspeed_entry': None,
        'maxspeed_exit': None,
        'oneway_entry': None,
        'oneway_exit': None
    },
    'label': {
        'track_line': None
    }
}

class Algorithm(object):
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
            track_line = test_sample['label']['track_line']
            intersection_angle = get_intersection_angle(entry_line, exit_line)
            curve_secant = get_curve_secant_line(entry_line, exit_line)
            track_radii = sample_line(curve_secant, track_line, intersection_angle)
            predicted_line = this_algorithm.predict(test_sample)
            radii =
