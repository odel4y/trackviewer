#!/usr/bin/python
#coding:utf-8
import numpy as np
import sklearn.preprocessing
import sklearn.ensemble
import automatic_test
import extract_features

class RandomForestAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self, features):
        self.name = 'Random Forest Regressor (Scikit)'
        for f in features:
            if f not in extract_features._feature_types:
                raise NotImplementedError("Random Forest Algorithm: Feature %s is not available" % f)
        self.name = self.name + '\nRegarded Features:\n- ' + '\n- '.join(features)
        self.features = features

    def train(self, samples):
        X, y = extract_features.get_matrices_from_samples(samples)
        X = self.filter_feature_matrix(X, self.features)
        print 'Training regressor with %d samples...' % (len(X))
        self.regressor = sklearn.ensemble.RandomForestRegressor()
        self.regressor.fit(X, y)

    def predict(self, sample):
        X, _ = extract_features.get_matrices_from_samples([sample])
        X = self.filter_feature_matrix(X, self.features)
        return self.regressor.predict(X)[0]

    def filter_feature_matrix(self, X, features):
        feature_indices = [extract_features._feature_types.index(f) for f in features]
        X_new = X[:,feature_indices]
        return X_new
