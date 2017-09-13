#!/usr/bin/python3
# coding=utf-8
"""
Comparison of various classification models over Credit Card Fraud Dectection dataset

Dataset source: https://www.kaggle.com/dalpozz/creditcardfraud
"""
from argparse import ArgumentParser
from os.path import isfile
from time import time
from urllib.request import urlretrieve

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utility import *


def main():
    # load and split data
    start = time()
    # check if file exists
    data_file = "data/creditcard.csv"
    if not isfile(data_file):
        try:
            # download the data set
            # Note: it is around 180MB
            data_url = "https://github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/raw/master/creditcard.csv"
            urlretrieve(data_url, data_file)
            print("download data file to %s" % data_file)
        except Error:
            print("can't access or download the data set")
            print("please try to download it manually and put into data/creditcard.csv")
            sys.exit()
    dataset, target = load_dataset(data_file)
    print("Loaded data in %.4f seconds" % (time() - start))
    start = time()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, target, test_size=.2, random_state=42)
    print("Training set size:%d, Testing set size: %d" %
          (len(x_train), len(x_test)))
    print("Prepared data for models in %.4f seconds" % (time() - start))
    scores = []
    models = {"GNB": GaussianNB(),
              "DT": DecisionTreeClassifier(max_depth=5),
              "MLP": MLPClassifier(alpha=1.0),
              #"LSVC": SVC(kernel="linear", C=0.025), # very slow as there is too much data
              "NN": KNeighborsClassifier(),
              "RF": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
              "ABC": AdaBoostClassifier(),
              "SGD": SGDClassifier(),
              }
    names = []
    for k, model in models.items():
        print("Running %s" % k)
        start = time()
        fitted_model = model.fit(x_train, y_train)
        print("Training time: %.4f seconds" % (time() - start))
        start = time()
        y_predicted = fitted_model.predict(x_test)
        print("Testing time: %.4f seconds" % (time() - start))
        scores.append(display(y_test, y_predicted,
                              save="figures/" + k + ".png"))
        names.append(k)
    # scatter plot scores of all the models
    plot_scores(scores, names, save="figures/scores.png")


main()
