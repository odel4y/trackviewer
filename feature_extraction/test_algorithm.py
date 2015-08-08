#!/usr/bin/python
#coding:utf-8
import sys
import pickle
from matplotlib import pyplot
import sklearn.preprocessing
import numpy as np

def convert_to_array(features, label):
    """Convert features to a number and put them in numpy array"""
    def convert_boolean(b):
        if b: return 1.0
        else: return -1.0
    label_len = len(label["angles"])
    feature_row = np.zeros((1,7))
    feature_row[0][0] = features["intersection_angle"]
    feature_row[0][1] = features["maxspeed_entry"]
    feature_row[0][2] = features["maxspeed_exit"]
    feature_row[0][3] = features["lane_distance_entry"]
    feature_row[0][4] = features["lane_distance_exit"]
    feature_row[0][5] = convert_boolean(features["oneway_entry"])
    feature_row[0][6] = convert_boolean(features["oneway_exit"])
    #a = numpy.matlib.repmat(feature_row, label_len, 1)
    #b = np.array(label["angles"])
    #feature_rows = np.column_stack((a,b))
    #label_rows = numpy.matlib.repmat(np.array(label["radii"]), label_len, 1)
    label_row = np.array(label["radii"])
    return feature_row, label_row

if __name__ == "__main__":
    with open('../data/models/model.pickle', 'rb') as f:
        reg = pickle.load(f)
    for fp in sys.argv[1:]:
        with open(fp, 'rb') as f:
            fc = pickle.load(f)
        features, label = fc
        X, y = convert_to_array(features, label)
        X = sklearn.preprocessing.normalize(X, axis=1, copy=False)
        yp = reg.predict(X)
        y_len = len(yp[0])
        fig = pyplot.figure()
        pyplot.hold(True)
        pyplot.plot(range(y_len), y, 'b.-')
        pyplot.plot(range(y_len), yp[0], 'r.-')
        pyplot.show()
