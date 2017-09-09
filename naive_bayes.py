#!/usr/bin/python3
# coding=utf-8

import argparse
import sys

from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB

from utility import *


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
    summaries = [(mean(attribute), stdev(attribute))
                 for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def separateByClass(dataset):
    '''
    Assumes the last column (attribute) to be class value
    '''
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probability = calculateProbability(x, mean, stdev)
            # ignore zero probability
            if probability != 0:
                probabilities[classValue] *= probability
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="file", type=str,
                        default="data/banknote.csv", help="data file")
    parser.add_argument('-s', '--split', dest='split', default=0.67, type=float,
                        help="data split ratio")
    return parser.parse_args()


def main():
    args = get_args()
    # load and split data
    trainingSet, testSet = split_dataset(load_dataset(args.file), args.split)
    print("Training set size: %d, Testing set size: %d" %
          (len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    actual = testSet[:, -1]
    display(actual, predictions)
    # using scikit
    gnb = GaussianNB()
    y_pred = gnb.fit(trainingSet[:, 0:-1],
                     trainingSet[:, -1]).predict(testSet[:, 0:-1])
    display(actual, y_pred)


main()
