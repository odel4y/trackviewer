#!/usr/bin/python
#coding:utf-8
import automatic_test
import numpy as np
import extract_features
import matplotlib.pyplot as plt
import seaborn as sns
import pandas

def get_array_from_feature(samples, feature):
    feature_i = extract_features._feature_types.index(feature)
    return np.array([s['X'][feature_i] for s in samples])

def plot_label_heatmap(samples, bars_y = 30):
    angles = np.linspace(0., np.pi, len(samples[0]['y']))
    min_x = np.amin(angles)
    max_x = np.amax(angles)
    radii = np.zeros((len(samples), len(angles)))
    for i, s in enumerate(samples):
        radii[i] = s['y']
    min_y = np.amin(radii)
    max_y = np.amax(radii)
    heatmap_array = np.zeros((bars_y, len(angles)))
    for i in xrange(np.shape(radii)[0]):
        for j in xrange(np.shape(radii)[1]):
            dest_j = j
            dest_i = round((radii[i,j] - min_y) / (max_y - min_y) * bars_y)-1
            heatmap_array[dest_i, dest_j] += 1
    indices = np.linspace(min_y, max_y, bars_y)
    indices = ["%.1f" % i for i in indices]
    columns = np.linspace(0., 180., len(angles))
    columns = ["%.1f" % i for i in columns]
    heatmap_array = np.flipud(heatmap_array)
    heatmap_frame = pandas.DataFrame(data=heatmap_array, index=reversed(indices), columns=columns)
    f = sns.heatmap(heatmap_frame)
    sns.plt.show(f)

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
intersection_angles = get_array_from_feature(samples, 'intersection_angle')/(np.pi)*180.0
#plot_histogram(intersection_angles, 16, 'Intersection Angles')
sns.set(color_codes=True)
plot_label_heatmap(samples)
angle_plot = sns.distplot(intersection_angles, bins=20, kde=False, rug=True)
sns.plt.show(angle_plot)
