#!/usr/bin/python
#coding:utf-8
from __future__ import division
from abc import ABCMeta, abstractmethod
from extract_features import get_intersection_angle, get_curve_secant_line,\
    sample_line, _feature_types, get_matrices_from_samples, get_samples_from_matrices,\
    get_predicted_line, _feature_types
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing
import random
import pickle
import itertools
import numpy as np
from plot_helper import plot_intersection, plot_graph

class PredictionAlgorithm(object):
    __metaclass__ = ABCMeta
    name = ''
    description = ''

    def train(self, samples):
        pass

    @abstractmethod
    def predict(self, sample):
        pass

    def get_name(self):
        return self.name

    def get_description(self):
        return self.description

def normalize_features(samples):
    """Normalize all the feature vectors in samples"""
    X, y = get_matrices_from_samples(samples)
    X = sklearn.preprocessing.normalize(X, axis=1, copy=False)
    return get_samples_from_matrices(X, y, samples)

def get_partitioned_samples(samples, train_ratio):
    """Randomize the given samples and partition them in train and test
    samples using the train_ratio"""
    sample_count = len(samples)
    print 'Total number of samples:', sample_count
    train_sample_count = int(round(sample_count * train_ratio))
    indices = range(sample_count)
    random.shuffle(indices)
    train_indices = indices[:train_sample_count]
    test_indices = indices[train_sample_count:]
    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]
    return train_samples, test_samples

def get_cross_validation_samples(samples, train_ratio, number, randomized=False):
    """Get several randomized train and test sets for cross validation"""
    sample_count = len(samples)
    print "Total number of samples:", sample_count
    train_sample_count = int(round(sample_count * train_ratio))
    indices = range(sample_count)
    if randomized == True:
        random.shuffle(indices)
    test_sample_count = sample_count - train_sample_count
    train_samples = []
    test_samples = []
    for i in range(number):
        i1 = int((sample_count - test_sample_count)/number)*i
        i2 = i1 + test_sample_count
        test_indices = indices[i1:i2]
        train_indices = indices[:i1] + indices[i2:]
        test_samples.append([samples[i] for i in test_indices])
        train_samples.append([samples[i] for i in train_indices])
    return train_samples, test_samples

def train(algorithms, train_samples):
    for algo in algorithms:
        algo.train(train_samples)

def predict(algorithms, test_samples):
    """Get a prediction for a trajectory from each algorithm"""
    results = {}
    for algo in algorithms:
        # Initialize result structure
        results[algo] = {
                            'mse': [],
                            'predictions': []
                        }
        for s in test_samples:
            y_true = s['y']
            y_pred = algo.predict(s)
            mse = mean_squared_error(y_true, y_pred)
            results[algo]['predictions'].append(y_pred)
            results[algo]['mse'].append(mse)
    return results

def predict_proba(algorithms, test_samples):
    """Predict probabilistic maps for algorithms that support this"""
    results_proba = {}
    for algo in algorithms:
        # Initialize result structure
        results_proba[algo] = {
                            'predictions_proba': [],
                            'bin_num': 0,
                            'min_radius': 0.,
                            'max_radius': 0.
                        }
        for s in test_samples:
            y_pred = algo.predict_proba_raw(s)
            results_proba[algo]['predictions_proba'].append(y_pred)
            results_proba[algo]['bin_num'] = algo.bin_num
            results_proba[algo]['min_radius'] = algo.min_radius
            results_proba[algo]['max_radius'] = algo.max_radius
    return results_proba

def show_intersection_plot(results, test_samples, results_proba={}, which_algorithms="all", which_samples="all", orientation="preserve"):
    print "Show intersection plot..."
    if which_algorithms == "all":
        which_algorithms = results.keys()
    plot_cases = {} # Contains the indices of the samples to be plotted and a plot title
    if which_samples == "all":
        plot_cases = {i:"Sample %d/%d" % (i, len(test_samples)) for i in range(len(test_samples))}
    if which_samples in ["best-case", "best-worst-case"]:
        for algo in which_algorithms:
            best_case_index = results[algo]['mse'].index(min(results[algo]['mse']))
            try:
                plot_cases[best_case_index] += " | Best case for " + algo.get_name()
            except:
                plot_cases[best_case_index] = "Best case for " + algo.get_name()
    if which_samples in ["worst-case", "best-worst-case"]:
        for algo in which_algorithms:
            worst_case_index = results[algo]['mse'].index(max(results[algo]['mse']))
            try:
                plot_cases[worst_case_index] += " | Worst case for " + algo.get_name()
            except:
                plot_cases[worst_case_index] = "Worst case for " + algo.get_name()
    for plot_index, plot_title in plot_cases.iteritems():
        predicted_radii = []
        predicted_proba = []
        labels = []
        s = test_samples[plot_index]
        for algo in which_algorithms:
            predicted_radii.append(results[algo]['predictions'][plot_index])
            labels.append(algo.get_name())
        for algo in results_proba:
            # Append heatmap data together with bin characterstics of algorithm
            predicted_proba.append({
                            'predictions_proba': results_proba[algo]['predictions_proba'][plot_index],
                            'bin_num': results_proba[algo]['bin_num'],
                            'min_radius': results_proba[algo]['min_radius'],
                            'max_radius': results_proba[algo]['max_radius']
                        })
        plot_intersection(s, predicted_radii, predicted_proba, labels, plot_title, orientation=orientation)

def show_graph_plot(results, test_samples, results_proba={}, which_algorithms="all", which_samples="all"):
    print "Show graph plot..."
    if which_algorithms == "all":
        which_algorithms = results.keys()
    plot_cases = {} # Contains the indices of the samples to be plotted and a plot title
    if which_samples == "all":
        plot_cases = {i:"Sample %d/%d" % (i, len(test_samples)) for i in range(len(test_samples))}
    if which_samples in ["best-case", "best-worst-case"]:
        for algo in which_algorithms:
            best_case_index = results[algo]['mse'].index(min(results[algo]['mse']))
            try:
                plot_cases[best_case_index] += " | Best case for " + algo.get_name()
            except:
                plot_cases[best_case_index] = "Best case for " + algo.get_name()
    if which_samples in ["worst-case", "best-worst-case"]:
        for algo in which_algorithms:
            worst_case_index = results[algo]['mse'].index(max(results[algo]['mse']))
            try:
                plot_cases[worst_case_index] += " | Worst case for " + algo.get_name()
            except:
                plot_cases[worst_case_index] = "Worst case for " + algo.get_name()
    for plot_index, plot_title in plot_cases.iteritems():
        track_coords = test_samples[plot_index]['y']
        predicted_coords = []
        labels = []
        s = test_samples[plot_index]
        predicted_proba = []
        for algo in which_algorithms:
            predicted_coords.append(results[algo]['predictions'][plot_index])
            labels.append(algo.get_name())
        for algo in results_proba:
            # Append heatmap data together with bin characterstics of algorithm
            predicted_proba.append({
                            'predictions_proba': results_proba[algo]['predictions_proba'][plot_index],
                            'bin_num': results_proba[algo]['bin_num'],
                            'min_radius': results_proba[algo]['min_radius'],
                            'max_radius': results_proba[algo]['max_radius']
                        })
        plot_graph(track_coords, predicted_coords, predicted_proba, labels, plot_title)

def get_result_statistics(results):
    """Return different statistic measures for the given results"""
    result_statistics = {}
    for algo, result in results.iteritems():
        result_statistics[algo] =   {
                                        'cumulated_mse': sum(result['mse']),
                                        'average_mse': np.mean(result['mse']),
                                        'std_mse': np.std(result['mse']),
                                        'min_mse': min(result['mse']),
                                        'max_mse': max(result['mse'])
                                    }
    return result_statistics

def output_formatted_result(results, output="console"):
    result_statistics = get_result_statistics(results)
    for algo, rs in result_statistics.iteritems():
        if output == "console":
            print 'Test with algorithm:', algo.get_name()
            if algo.get_description() != '':
                print 'Description:', algo.get_description()
            print 'Average MSE:\t%.2f +/- %.2f' % (rs['average_mse'], rs['std_mse'])
            print 'Cumulated MSE:\t%.2f' % rs['cumulated_mse']
            print 'Minimum MSE:\t%.2f' % rs['min_mse']
            print 'Maximum MSE:\t%.2f' % rs['max_mse']

def test(algorithms, train_sample_sets, test_sample_sets, cross_validation=False):
    """General prediction quality test for algorithms with the option of cross validation"""
    results = []
    if not cross_validation:
        train_sample_sets = [train_sample_sets]
        test_sample_sets = [test_sample_sets]
    for train_samples, test_samples in zip(train_sample_sets, test_sample_sets):
        train(algorithms, train_samples)
        results.append(predict(algorithms, test_samples))
    flattened_results = results[0]
    for result in results[1:]:
        for algo in result:
            flattened_results[algo]['predictions'].extend(result[algo]['predictions'])
            flattened_results[algo]['mse'].extend(result[algo]['mse'])
    output_formatted_result(flattened_results)

def test_feature_permutations(algo_class, train_sample_sets, test_sample_sets, features, min_num_features=4, cross_validation=False):
    """Test the prediction quality of all possible combinations of features with
    a minimum number of features with a certain Algorithm class. Cross validation is possible"""
    feature_sets = []               # Contains all the possible combinations of features with a minimum number of them
    results = []
    rating_arg = 'average_mse'      # The argument to be used as a comparison score
    feature_quality_dicts = []
    if not cross_validation:
        train_sample_sets = [train_sample_sets]
        test_sample_sets = [test_sample_sets]
    # Build the possible feature combinations
    for num_features in range(min_num_features, len(features)):
        it = itertools.combinations(features, num_features)
        it_list = list(it)
        feature_sets.extend(it_list)
    # When cross validating iterate over the different sample sets
    for train_samples, test_samples in zip(train_sample_sets, test_sample_sets):
        print "Testing %d different feature sets" % len(feature_sets)
        # Test all the feature combinations with the given algorithm
        for i, f_set in enumerate(feature_sets):
            algo = algo_class(f_set)
            train([algo], train_samples)
            result = predict([algo], test_samples)
            rs = get_result_statistics(result)
            print "Algorithm %d/%d has quality: %.2f" % (i+1, len(feature_sets), rs[algo][rating_arg])
            results.append((algo, rs[algo]))
        feature_occurences = {}
        for res in results:
            features = res[0].features
            for f in features:
                if f in feature_occurences:
                    feature_occurences[f].append(res[1][rating_arg])
                else:
                    feature_occurences[f] = [res[1][rating_arg]]
        feature_quality = {}
        for k,v in feature_occurences.iteritems():
            feature_quality[k] = sum(v)/len(v)
        feature_quality_dicts.append(feature_quality)
    total_feature_quality = {}
    for f in feature_quality_dicts[0].keys():
        f_scores = [res[f] for res in feature_quality_dicts]
        total_feature_quality[f] = sum(f_scores)/len(f_scores)
    sorted_feature_importance = sorted(total_feature_quality.items(), key=lambda it: it[1])
    print "====== FEATURE QUALITY ======"
    for i, (f, score) in enumerate(sorted_feature_importance):
        print "#%d %s: %.2f" % (i+1, f, score)

def load_samples(fn):
    with open(fn, 'r') as f:
        samples = pickle.load(f)
    return samples
