#!/usr/bin/python
#coding:utf-8
import sys
import pickle
import numpy as np
import numpy.matlib
import os.path
import sklearn.preprocessing
import sklearn.ensemble

# _features = {
#     "intersection_angle": None,
#     "maxspeed_entry": None,
#     "maxspeed_exit": None,
#     "lane_distance_entry": None,
#     "lane_distance_exit": None,
#     "oneway_entry": None,
#     "oneway_exit": None
# }
#
# _label = {
#     "angles": None
#     "radii": None
# }

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
    X = None
    y = None
    d,_ = os.path.split(sys.argv[0])
    for fp in sys.argv[1:]:
        fp = os.path.abspath(fp)
        with open(fp, 'rb') as f:
            fc = pickle.load(f)
            print 'Opening', fp
            features, label = fc
            feature_rows, label_rows = convert_to_array(features, label)
            if X == None and y == None:
                X = feature_rows
                y = label_rows
            else:
                X = np.vstack((X, feature_rows))
                y = np.vstack((y, label_rows))
    # Normalize all the feature data (Normalize along features?)
    print 'Feature normalizing...'
    X = sklearn.preprocessing.normalize(X, axis=1, copy=False)
    reg = sklearn.ensemble.RandomForestRegressor()
    print 'Training...'
    reg.fit(X,y)
    print 'Saving model...'
    with open(os.path.join(d,'..','data','models','model.pickle'), 'wb') as f:
        pickle.dump(reg, f)
