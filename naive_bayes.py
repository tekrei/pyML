#!/usr/bin/python3
# coding=utf-8

from argparse import ArgumentParser
from math import pi as PI
from math import exp, sqrt

from numpy import mean, std
from sklearn.naive_bayes import GaussianNB

from utility import display, load_dataset, split_dataset


'''
naive bayes implementation

It is tested with the following example data sets:
- pima-indians-diabetes
- lung-cancer
- forestfires
- phishing_websites
- arrhythmia
- banknote

The main source for the code is the following tutorial: http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
'''


def summarize(dataset):
    summaries = [(mean(attribute), std(attribute))
                 for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def separate_by_class(dataset, target):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        result = target[i]
        if (result not in separated):
            separated[result] = []
        separated[result].append(vector)
    return separated


def summarize_by_class(dataset, target):
    separated = separate_by_class(dataset, target)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculate_probability(x, mean, stdev):
    '''
    Gaussian Probability Density Function calculation
    '''
    if mean == 0 or stdev == 0:
        return 0
    exponent = exp(-(pow(x - mean, 2) / (2 * pow(stdev, 2))))
    return (1 / (sqrt(2 * PI) * stdev)) * exponent


def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probability = calculate_probability(x, mean, stdev)
            # ignore zero probability
            if probability != 0:
                probabilities[class_value] *= probability
    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="file", type=str,
                        default="data/banknote.csv", help="data file")
    parser.add_argument('-s', '--split', dest='split', default=0.67, type=float,
                        help="data split ratio")
    return parser.parse_args()


def main():
    args = get_args()
    # load and split data
    dataset, target = load_dataset(args.file)
    train_x, train_y, test_x, actual = split_dataset(
        dataset, target, args.split)
    print("Training set size: %d, Testing set size: %d" %
          (len(train_x), len(test_x)))
    # prepare model
    summaries = summarize_by_class(train_x, train_y)
    # test model
    predictions = get_predictions(summaries, test_x)
    display(actual, predictions)
    # using scikit
    gnb = GaussianNB()
    y_pred = gnb.fit(train_x, train_y).predict(test_x)
    display(actual, y_pred)


main()
