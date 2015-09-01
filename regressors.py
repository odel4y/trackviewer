#!/usr/bin/python
#coding:utf-8
import numpy as np
import sklearn.preprocessing
import sklearn.ensemble
import automatic_test
from extract_features import _feature_types

class RandomForestAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self, features):
        self.name = 'Random Forest Regressor (Scikit)'
        for feature in features:
            if feature not in _feature_types:
                raise NotImplementedError("Random Forest Algorithm: Feature %s is not available" % feature)
        self.name = self.name + '\nRegarded Features:\n' + '\n- '.join(features)
        self.features = features

    def train(self, samples):
        X, y = self.convert_samples_to_arrays(samples)
        # TODO: Normalisieren aller Samples am Anfang
        X = sklearn.preprocessing.normalize(X, axis=1, copy=False)
        print 'Training regressor with %d samples...' % (len(X))
        self.regressor = sklearn.ensemble.RandomForestRegressor()
        self.regressor.fit(X, y)

    def predict(self, samples):
        X, _ = self.convert_samples_to_arrays(samples)
        # TODO: Normalisieren aller Samples am Anfang
        X = sklearn.preprocessing.normalize(X, axis=1, copy=False)
        return self.regressor.predict(X)

    def convert_samples_to_arrays(self, samples):
        """Convert the samples to numpy arrays for training or testing while only
        regarding the selected features"""
        label_len = len(samples[0]['y'])
        X = np.zeros((len(samples), len(self.features)))
        y = np.zeros((len(samples), label_len))
        for i, sample in enumerate(samples):
            X_row_filtered = [sample['X'][_feature_types.index(feature)] for feature in self.features]
            X[i] = np.array(X_row_filtered)
            y[i] = np.array(sample['y'])
        return X, y
