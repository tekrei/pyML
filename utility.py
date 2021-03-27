#     utility.py belongs to pyML
#
#     Copyright 2021 T. E. Kalayci
#     SPDX-License-Identifier: Apache-2.0
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

#     utility.py belongs to pyML
#
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import gzip
import pickle
import sys
from itertools import product
from os.path import isfile
from random import random
from urllib.request import urlretrieve

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize


def sigmoid(x):
    """ the sigmoid function """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(z):
    """ derivative of the sigmoid function """
    return sigmoid(z) * (1 - sigmoid(z))


def euclidean(x, y):
    return sum(np.sqrt((x - y) ** 2))


def check_and_download(data_file, data_url):
    if not isfile(data_file):
        print(f"{data_file} doesn't exists, will download")
        try:
            urlretrieve(data_url, data_file)
            print(f"Downloaded data file to {data_file}")
        except:
            print(f"Can't access or download the data set. Please try to download it manually and put into {data_file}")
            sys.exit()


def load_dataset(data_file, split=True, binarize=False):
    dataset = pd.read_csv(data_file, sep=",").sample(
        frac=1).reset_index(drop=True)
    if not split:
        return np.asarray(dataset)
    target = dataset.iloc[:, -1]
    if binarize:
        target = binarize_labels(target)
    dataset = np.asarray(dataset.iloc[:, 0:-1])
    return dataset, target


def get_accuracy(actual, predictions):
    total_count = len(actual)
    wrong_count = (actual != predictions).sum()
    print("Number of mislabeled points out of a total %d points : %d" % (
        total_count, wrong_count))
    return (total_count - wrong_count) / total_count * 100.0


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
    scores = np.asarray(scores)
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


def plot_confusion_matrix(actual, predictions, save=False, normalize=False, title='Confusion matrix',
                          cmap=plot.cm.get_cmap("Blues")):
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
    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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
    return np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    data_file = "./data/mnist.pkl.gz"
    # Note: it is around 17MB
    check_and_download(data_file,
                       "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz")
    f = gzip.open(data_file, 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data


def prepare_data():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return list(training_data), list(validation_data), list(test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
