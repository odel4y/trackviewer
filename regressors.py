#!/usr/bin/python
#coding:utf-8
from __future__ import division
import numpy as np
import sklearn.preprocessing
import sklearn.ensemble
from sklearn.externals.joblib import Parallel, delayed
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.utils.validation import DataConversionWarning, check_is_fitted
from sklearn.ensemble.base import BaseEnsemble, _partition_estimators
from scipy.sparse import issparse
from sklearn.ensemble.forest import _parallel_helper
import automatic_test
import extract_features

def filter_feature_matrix(X, features):
    feature_indices = [extract_features._feature_types.index(f) for f in features]
    X_new = X[:,feature_indices]
    return X_new

def _check_feature_availability(features):
    for f in features:
        if f not in extract_features._feature_types:
            raise NotImplementedError("Feature %s is not available" % f)

class RandomForestAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self, features, n_estimators=10):
        self.name = 'Random Forest Regressor (Scikit)'
        _check_feature_availability(features)

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

    def predict_all_estimators(self, sample):
        """Get the prediction of every estimator separated"""
        X, _ = extract_features.get_matrices_from_samples([sample])
        X = filter_feature_matrix(X, self.features)
        # Most of the code is directly copied from Scikit
        # Check data
        check_is_fitted(self.regressor, 'n_outputs_')

        # Check data
        X = check_array(X, dtype=DTYPE, accept_sparse="csr")
        if issparse(X) and (X.indices.dtype != np.intc or
                            X.indptr.dtype != np.intc):
            raise ValueError("No support for np.int64 index based "
                             "sparse matrices")

        # Assign chunk of trees to jobs
        n_jobs, n_trees, starts = _partition_estimators(self.regressor.n_estimators,
                                                        self.regressor.n_jobs)

        # Parallel loop
        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.regressor.verbose,
                             backend="threading")(
            delayed(_parallel_helper)(e, 'predict', X, check_input=False)
            for e in self.regressor.estimators_)

        return all_y_hat

class RFClassificationAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self, features, bin_num, min_radius, max_radius, n_estimators):
        self.name = 'Random Forest Classifier (Scikit)'
        _check_feature_availability(features)
        self.features = features
        self.bin_num = bin_num
        self.classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators)

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
        _check_feature_availability(features)
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

class RandomForestSeparatedDirectionsAlgorithm(automatic_test.PredictionAlgorithm):
    """Use the ordinary RandomForestRegressor but train separate predictors for different intersection angles"""
    def __init__(self, features, n_estimators=10):
        self.name = 'Random Forest Regressor (Scikit) with separate predictors for different intersection angles'
        _check_feature_availability(features)

        self.description = 'Regarded Features:\n- ' + '\n- '.join(features)
        self.features = features
        self.n_estimators = n_estimators

    def _is_left_turn(self, sample):
        return sample['X'][extract_features._feature_types.index('intersection_angle')] >= 0.0

    def train(self, samples):
        samples_l = [s for s in samples if self._is_left_turn(s)]
        samples_r = [s for s in samples if not self._is_left_turn(s)]

        X_l, y_l = extract_features.get_matrices_from_samples(samples_l)
        X_r, y_r = extract_features.get_matrices_from_samples(samples_r)
        X_l = filter_feature_matrix(X_l, self.features)
        X_r = filter_feature_matrix(X_r, self.features)

        self.regressor_l = sklearn.ensemble.RandomForestRegressor(n_estimators=int(self.n_estimators/2))
        self.regressor_r = sklearn.ensemble.RandomForestRegressor(n_estimators=int(self.n_estimators/2))
        self.regressor_l.fit(X_l, y_l)
        self.regressor_r.fit(X_r, y_r)

    def predict(self, sample):
        X, _ = extract_features.get_matrices_from_samples([sample])
        X = filter_feature_matrix(X, self.features)
        if self._is_left_turn(sample):
            return self.regressor_l.predict(X)[0]
        else:
            return self.regressor_r.predict(X)[0]
