from itertools import product
from random import random

import matplotlib.pyplot as plot
from numpy import arange, asarray, newaxis, sqrt
from pandas import read_csv
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize


def euclidean(A, B):
    return sum(sqrt((A - B)**2))


def load_dataset(data_file, split=True, binarize=False):
    dataset = read_csv(data_file, sep=",").sample(
        frac=1).reset_index(drop=True)
    if not split:
        return asarray(dataset)
    target = dataset.iloc[:, -1]
    if binarize:
        target = binarize_labels(target)
    dataset = asarray(dataset.iloc[:, 0:-1])
    return dataset, target


def get_accuracy(actual, predictions):
    totalCount = len(actual)
    wrongCount = (actual != predictions).sum()
    print("Number of mislabeled points out of a total %d points : %d" % (
        totalCount, wrongCount))
    return (totalCount - wrongCount) / totalCount * 100.0


def display(actual, predictions, save=None):
    print(f"Accuracy: {accuracy_score(actual, predictions)}")
    print(f"Precision: {precision_score(actual, predictions, average='weighted')}")
    print(f"Recall: {recall_score(actual, predictions, average='weighted')}")
    f1 = f1_score(actual, predictions, average="weighted")
    print(f"F1 score: {f1}")
    roc = roc_auc(actual, predictions)
    print(f"ROC AUC Score: {roc}")
    # plot non-normalized confusion matrix
    plot_confusion_matrix(actual, predictions, save)
    return [f1, roc]


def plot_scores(scores, names, save=None):
    scores = asarray(scores)
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


def binarize_labels(actual):
    return label_binarize(actual, list(set(actual)))


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
    tick_marks = arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]

    thresh = cm.max() / 2.0
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, cm[i, j], horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    if save:
        plot.savefig(save)
    else:
        plot.show()
    return cm.ravel()


def split_dataset(dataset, target, split_factor):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for x in range(len(dataset) - 1):
        if random() < split_factor:
            train_x.append(dataset[x])
            train_y.append(target[x])
        else:
            test_x.append(dataset[x])
            test_y.append(target[x])
    return asarray(train_x), asarray(train_y), asarray(test_x), asarray(test_y)
