#!/usr/bin/python
#coding:utf-8
# Create synthetic samples of different intersection angles
from __future__ import division
import sys
sys.path.append('../')
from create_synthetic import create_intersection_data
import numpy as np
import regressors
import automatic_test
from shapely.geometry import LineString, Point
from extract_features import create_sample, upsample_line

tags = {
    "entry_way": {
        "highway": "secondary",
        "maxspeed": "30",
        "name": "Heinheimer Stra\u00dfe"
    },
    "exit_way": {
        "highway": "residential",
        "maxspeed": "30",
        "name": "Vogelsbergstra\u00dfe"
    }
}


# Create number of LineString serving as way data
angles = np.linspace(-3*np.pi/8, 11*np.pi/8, 15)
scale = 50.0
coords = [[(0,0),(0,1*scale),(np.cos(a)*scale,(1+np.sin(a))*scale)] for a in angles]
labels = ["Test way %.2f°" % (a/np.pi*180.) for a in angles]
lines = [LineString(c) for c in coords]
test_samples = []
# Create synthetic samples
for line, label in zip(lines, labels):
    int_sit, osm = create_intersection_data(line, 0.5, normalized=True, tags=tags)
    sample = create_sample(int_sit, osm, label, output="console")
    test_samples.append(sample)
train_samples = automatic_test.load_samples('../data/training_data/samples_23_09_15/samples.pickle')
# train_samples = automatic_test.normalize_features(samples)
# Load algorithm with train samples and test the synthetic ones
rf_algo = regressors.RandomForestAlgorithm()
automatic_test.train([rf_algo], train_samples)
results = automatic_test.predict([rf_algo], test_samples)
automatic_test.show_intersection_plot(results, test_samples, orientation="curve-secant")
