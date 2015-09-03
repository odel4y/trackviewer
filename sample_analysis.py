#!/usr/bin/python
#coding:utf-8
import automatic_test
import numpy as np
import extract_features
import matplotlib.pyplot as plt

def get_array_from_feature(samples, feature):
    feature_i = extract_features._feature_types.index(feature)
    return np.array([s['X'][feature_i] for s in samples])

def plot_histogram(array, bins, title, block=True):
    hist, bins = np.histogram(array, bins=bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title(title)
    plt.show(block=block)

samples = automatic_test.load_samples('data/training_data/samples.pickle')
print 'Sample count:', len(samples)
oneway_entry = list(get_array_from_feature(samples, 'oneway_entry'))
print 'oneway_entry: Yes: %d No: %d' % (oneway_entry.count(1.0), oneway_entry.count(-1.0))
oneway_exit = list(get_array_from_feature(samples, 'oneway_exit'))
print 'oneway_exit: Yes: %d No: %d' % (oneway_exit.count(1.0), oneway_exit.count(-1.0))
intersection_angles = get_array_from_feature(samples, 'intersection_angle')/(np.pi*2)*180.0
plot_histogram(intersection_angles, 16, 'Intersection Angles')
