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

    def train(self, samples):
        pass

    def predict(self, samples):
        pass
