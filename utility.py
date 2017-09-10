import csv
import itertools
import math
import random

import matplotlib.pyplot as plot
import numpy
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize


def euclidean(A, B):
    return numpy.sqrt(((A - B)**2).sum())


def load_dataset(filename):
    with open(filename, 'rt') as csvfile:
        # all values without quotes are considered numeric
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        return list(lines)


def load_numbers(filename, delimiter=","):
    return numpy.genfromtxt(filename, delimiter=delimiter)


def mean(numbers):
    return numpy.mean(numbers)


def stdev(numbers):
    return numpy.std(numbers)


def f1_manual(actual, predictions, average='weighted'):
    p = precision(actual, predictions, average)
    r = recall(actual, predictions, average)
    return 2 * ((p * r) / (p + r))


def f1_library(actual, predictions, average='weighted'):
    return f1_score(actual, predictions, average=average)


def recall(actual, predictions, average='weighted'):
    return recall_score(actual, predictions, average=average)


def precision(actual, predictions, average='weighted'):
    return precision_score(actual, predictions, average=average)


def accuracy(actual, predictions):
    return accuracy_score(actual, predictions)


def getAccuracy(actual, predictions):
    totalCount = len(actual)
    wrongCount = (actual != predictions).sum()
    print("Number of mislabeled points out of a total %d points : %d" % (
        totalCount, wrongCount))
    return (totalCount - wrongCount) / totalCount * 100.0


def display(actual, predictions, save=None):
    print("Accuracy: %f" % accuracy(actual, predictions))
    print("Precision: %f" % precision(actual, predictions))
    r = recall(actual, predictions)
    print("Recall: %f" % r)
    #print("F1 score (manual):%f" % f1_manual(actual, predictions))
    f1 = f1_library(actual, predictions)
    print("F1 score (scikit):%f" % f1)
    roc = roc_auc(actual, predictions)
    print("ROC AUC Score:%f" % roc)
    # plot non-normalized confusion matrix
    plot_confusion_matrix(actual, predictions, save)
    return [f1, roc]


def plot_scores(scores, names, save=None):
    scores = numpy.asarray(scores)
    # roc_auc and f1 plot
    plot.figure()
    plot.scatter(scores[:, 0], scores[:, 1])
    for i, txt in enumerate(names):
        plot.annotate(txt, (scores[i, 0], scores[i, 1]))
    plot.xlim(xmax=1.0)
    plot.ylim(ymax=1.0)
    plot.ylabel('ROC AUC Score')
    plot.xlabel('F1 score')
    if save:
        plot.savefig(save)
    else:
        plot.show()


def roc_auc(actual, predictions, average='weighted'):
    class_names = list(set(actual))
    # use binarized values for AUC score calculation
    return roc_auc_score(label_binarize(actual, class_names), label_binarize(predictions, class_names), average=average)


def plot_confusion_matrix(actual, predictions, save=False, normalize=False, title='Confusion matrix', cmap=plot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    cm = confusion_matrix(actual, predictions)
    classes = list(set(actual))
    plot.figure()
    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = numpy.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, cm[i, j], horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    if save:
        plot.savefig(save)
    else:
        plot.show()
    return cm.ravel()


def calculateProbability(x, mean, stdev):
    '''
    Gaussian Probability Density Function calculation
    '''
    if mean == 0 or stdev == 0:
        return 0
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def split_dataset(dataset, splitFactor):
    trainingSet = []
    testSet = []
    for x in range(len(dataset) - 1):
        if random.random() < splitFactor:
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])
    return numpy.asarray(trainingSet), numpy.asarray(testSet)
