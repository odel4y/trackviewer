#!/usr/bin/python
#coding:utf-8
from __future__ import division
import numpy as np
import sklearn.preprocessing
import sklearn.ensemble
import automatic_test
import extract_features

def filter_feature_matrix(X, features):
    feature_indices = [extract_features._feature_types.index(f) for f in features]
    X_new = X[:,feature_indices]
    return X_new

class RandomForestAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self, features):
        self.name = 'Random Forest Regressor (Scikit)'
        for f in features:
            if f not in extract_features._feature_types:
                raise NotImplementedError("Random Forest Algorithm: Feature %s is not available" % f)
        self.description = 'Regarded Features:\n- ' + '\n- '.join(features)
        self.features = features

    def train(self, samples):
        X, y = extract_features.get_matrices_from_samples(samples)
        X = filter_feature_matrix(X, self.features)
        print 'Training regressor with %d samples...' % (len(X))
        self.regressor = sklearn.ensemble.RandomForestRegressor()
        self.regressor.fit(X, y)

    def predict(self, sample):
        X, _ = extract_features.get_matrices_from_samples([sample])
        X = filter_feature_matrix(X, self.features)
        return self.regressor.predict(X)[0]

class RFClassificationAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self, features, bin_num, min_radius, max_radius):
        self.name = 'Random Forest Classifier (Scikit)'
        self.features = features
        self.bin_num = bin_num
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.classifier = sklearn.ensemble.RandomForestClassifier()

    def train(self, samples):
        X, y = extract_features.get_matrices_from_samples(samples)
        X = filter_feature_matrix(X, self.features)
        print 'Training classifier with %d samples...' % (len(X))
        self.angle_steps = np.linspace(0.,180.,np.shape(y)[1])
        steps = len(self.angle_steps)
        X_new = np.zeros((np.shape(X)[0]*steps, np.shape(X)[1]+1))
        # Introduce the angle as new feature
        for i, angle in enumerate(self.angle_steps):
            X_new[i*steps:(i+1)*steps, :-1] = np.tile(X[i], (steps, 1))
            X_new[i*steps:(i+1)*steps, -1] = self.angle_steps
        X = X_new
        y = np.ravel(y)
        self.classifier.fit(X, y)

    def predict(self, sample):
        X, _ = extract_features.get_matrices_from_samples([sample])
        X = filter_feature_matrix(X, self.features)
        X = np.tile(X, (len(self.angle_steps), 1))
        X = np.column_stack((X, self.angle_steps))
        y_pred = np.ravel(self.classifier.predict(X))
        return self.bin_to_continuous(y_pred)

    def predict_proba_raw(self, sample):
        X, _ = extract_features.get_matrices_from_samples([sample])
        X = filter_feature_matrix(X, self.features)
        y_pred = self.classifier.predict_proba(X)[0]
        return y_pred

    def continuous_to_bin(self, v):
        return np.floor((v - self.min_radius) / (self.max_radius - self.min_radius) * self.bin_num)

    def bin_to_continuous(self, v):
        return self.min_radius + (self.max_radius - self.min_radius) * (v / self.bin_num)
