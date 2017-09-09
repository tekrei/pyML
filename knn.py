#!/usr/bin/python3
# coding=utf-8
import argparse
import operator
import sys

import numpy

from utility import *


'''
k-Nearest Neighbors implementation

- Doesn't use any library to perform KNN.
- Uses scikit-learn library for calculating various metrics and confusion matrix.

It is possible to provide file name, k value and training-test data split ratio as arguments such as the following:
        python knn.py data/iris.csv 5 0.67

It is tested with the following example data sets:
- iris: categorical result value are converted to numeric values (https://archive.ics.uci.edu/ml/datasets/Iris)
- forestfires: categorical values (mon, day) are converted to numeric values, all values larger than 0 are converted to 1 in burned area column (https://archive.ics.uci.edu/ml/datasets/Forest+Fires)
- lung-cancer: moved target values to the last column, missed values replaced by -1 (https://archive.ics.uci.edu/ml/datasets/Lung+Cancer)
- phishing_websites: nothing changed, converted to CSV without header (https://archive.ics.uci.edu/ml/datasets/Phishing+Websites)
- arrhythmia: missed values replaced by -1 (https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
- banknote: nothing changed, converted to CSV (https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

The main source for the code is the following tutorial: Source: http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
'''


def get_neighbors(training, test, k):
    distances = []
    for x in range(len(training)):
        # without target
        dist = euclidean(test[0:-1], training[x, 0:-1])
        distances.append((training[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(),
                          key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="file", type=str,
                        default="data/iris.csv", help="data file")
    parser.add_argument('-k', dest='k', default=5, type=int,
                        help="number of neighbors to consider")
    parser.add_argument('-s', '--split', dest='split', default=0.67, type=float,
                        help="data split ratio")
    return parser.parse_args()


def main():
    args = get_args()
    # load data
    training, test = split_dataset(load_dataset(args.file), args.split)
    print("Training set size: %d" % (len(training)))
    print("Testing set size: %d" % (len(test)))
    # generate predictions
    predictions = []
    actual = []
    for x in range(len(test)):
        neighbors = get_neighbors(training, test[x], args.k)
        result = get_response(neighbors)
        predictions.append(result)
        actual.append(test[x][-1])
    # calculate and display various scores
    display(actual, predictions)


main()
