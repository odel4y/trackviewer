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
    def __init__(self, features, n_estimators=10):
        self.name = 'Random Forest Regressor (Scikit)'
        for f in features:
            if f not in extract_features._feature_types:
                raise NotImplementedError("Random Forest Algorithm: Feature %s is not available" % f)
        self.description = 'Regarded Features:\n- ' + '\n- '.join(features)
        self.features = features
        self.n_estimators = n_estimators

    def train(self, samples):
        X, y = extract_features.get_matrices_from_samples(samples)
        X = filter_feature_matrix(X, self.features)
        self.regressor = sklearn.ensemble.RandomForestRegressor(n_estimators=self.n_estimators)
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
        self.min_radius = np.amin(y)
        self.max_radius = np.amax(y)
        y = self.continuous_to_bin(np.ravel(y))
        self.classifier.fit(X, y)

    def predict(self, sample):
        X, _ = extract_features.get_matrices_from_samples([sample])
        X = filter_feature_matrix(X, self.features)
        X = np.tile(X, (len(self.angle_steps), 1))
        X = np.column_stack((X, self.angle_steps))
        y_pred = np.ravel(self.classifier.predict(X))
        return self.bin_to_continuous(y_pred)

    def predict_proba_raw(self, sample):
        """Return the raw output of predicting the probability for each class"""
        X, _ = extract_features.get_matrices_from_samples([sample])
        X = filter_feature_matrix(X, self.features)
        X = np.tile(X, (len(self.angle_steps), 1))
        X = np.column_stack((X, self.angle_steps))

        y_pred = self.classifier.predict_proba(X)

        # Pad the probability array with zero columns for disregarded classes
        missing_classes = [i for i in range(self.bin_num) if i not in self.classifier.classes_]
        y_pred = np.insert(y_pred, sorted(list(missing_classes)), 0., axis=1)

        return y_pred

    def continuous_to_bin(self, v):
        return np.minimum(
            np.floor((v - self.min_radius) / (self.max_radius - self.min_radius) * self.bin_num),
            np.ones(np.shape(v))*(self.bin_num-1)
            )

    def bin_to_continuous(self, v):
        half_bin_height = (self.max_radius - self.min_radius) / self.bin_num / 2
        return self.min_radius + (self.max_radius - self.min_radius) * (v / self.bin_num) + half_bin_height

class ExtraTreesAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self, features, n_estimators=10):
        self.name = 'Extra Trees Regressor (Scikit)'
        for f in features:
            if f not in extract_features._feature_types:
                raise NotImplementedError("Random Forest Algorithm: Feature %s is not available" % f)
        self.description = 'Regarded Features:\n- ' + '\n- '.join(features)
        self.features = features
        self.n_estimators = n_estimators

    def train(self, samples):
        X, y = extract_features.get_matrices_from_samples(samples)
        X = filter_feature_matrix(X, self.features)
        self.regressor = sklearn.ensemble.RandomForestRegressor(n_estimators=self.n_estimators)
        self.regressor.fit(X, y)

    def predict(self, sample):
        X, _ = extract_features.get_matrices_from_samples([sample])
        X = filter_feature_matrix(X, self.features)
        return self.regressor.predict(X)[0]
