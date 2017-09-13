#!/usr/bin/python3
# coding=utf-8
from argparse import ArgumentParser
from operator import itemgetter

from utility import display, euclidean, load_dataset, split_dataset


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
    distances = {}
    for x in range(len(training)):
        dist = euclidean(test, training[x])
        distances[x] = dist
    distances = sorted(distances.items(), key=itemgetter(1))
    neighbors = []
    for _ in range(k):
        neighbors.append(distances.pop()[0])
    return neighbors


def get_response(neighbors, target):
    class_votes = {}
    for x in neighbors:
        response = target[x]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(),
                          key=itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="file", type=str,
                        default="data/forestfires.csv", help="data file")
    parser.add_argument('-k', dest='k', default=5, type=int,
                        help="number of neighbors to consider")
    parser.add_argument('-s', '--split', dest='split', default=0.8, type=float,
                        help="data split ratio")
    return parser.parse_args()


def main():
    args = get_args()
    # load data
    dataset, target = load_dataset(args.file)
    train_x, train_y, test_x, test_y = split_dataset(
        dataset, target, args.split)
    print("Training set size: %d" % (len(train_x)))
    print("Testing set size: %d" % (len(test_x)))
    # generate predictions
    predictions = []
    actual = []
    for x in range(len(test_x)):
        neighbors = get_neighbors(train_x, test_x[x], args.k)
        result = get_response(neighbors, train_y)
        predictions.append(result)
        actual.append(test_y[x])
    # calculate and display various scores
    display(actual, predictions)


main()
