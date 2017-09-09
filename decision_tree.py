#!/usr/bin/python3
# coding=utf-8
"""
Decision tree with cross validation

It is tested with the following example data sets:
- banknote
- lung-cancer
- pima-indians-diabetes
- forestfires
- phishing_websites Note: taking long time
- arrhythmia        Note: taking long time

Source: http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
"""

import argparse
from random import randrange, seed

from sklearn.metrics import *

from utility import *


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        scores.append(accuracy_score(actual, predicted))
    return scores


def test_split(index, value, dataset):
    """ split a dataset based on an attribute and an attribute value """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def gini_index(groups, class_values):
    """ calculate the Gini index for a split dataset """
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1]
                          for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


def get_split(dataset):
    """ select the best split point for a dataset """
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):
    """ create a terminal node value """
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    """ create child splits for a node or make terminal """
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    """ Build a decision tree """
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    """ make a prediction with a decision tree """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    return [predict(tree, row) for row in test]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="file", type=str,
                        default="data/banknote.csv", help="data file")
    parser.add_argument('-n', '--nfolds', dest='n_folds', default=5, type=int,
                        help="number of folds")
    parser.add_argument('-d', '--depth', dest='max_depth', default=5, type=int,
                        help="maximum depth of a tree")
    parser.add_argument('-s', '--size', dest='min_size', default=10, type=int,
                        help="minimum size of a tree")
    return parser.parse_args()


def main():
    seed(1)
    args = get_args()
    # load and prepare data
    dataset = load_dataset(args.file)
    # dataset, algorithm to run, number of folds, maximum depth, minimum size
    scores = evaluate_algorithm(
        dataset, decision_tree, args.n_folds, args.max_depth, args.min_size)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f' % (sum(scores) / float(len(scores))))


if __name__ == "__main__":
    main()
