#!/usr/bin/python3
# coding=utf-8
"""
Neural Network demonstration with Backpropagation learning and the Iris dataset
by M. Tim Jones from https://github.com/mtimjones/backprop

Python code by tekrei
"""

from argparse import ArgumentParser
from math import exp
from random import randrange

from numpy import ones, zeros
from numpy.random import rand

from utility import load_dataset


def sigmoid(x): return (1.0 / (1.0 + exp(-x)))


def sigmoid_(x): return (x * (1.0 - x))


class Backpropagation:

    def __init__(self, dataset, target, n_hidden=25):
        self.dataset = dataset
        self.count = len(self.dataset)
        n_input = len(self.dataset[0])
        self.target = target
        n_output = len(self.target[0])
        # Neuron cell values
        self.inputs = ones(n_input + 1)
        self.hidden = ones(n_hidden + 1)
        self.outputs = zeros(n_output)
        # initialize the network with random weights
        # weights are randomly chosen between 0-0.5
        self.weights_hidden_input = rand(n_hidden, n_input + 1) / 2
        self.weights_output_hidden = rand(n_output, n_hidden + 1) / 2

    def feed_forward(self, data):
        # given the test input, feed forward to the output
        self.inputs = data
        # calculate hidden layer outputs
        for i in range(len(self.hidden) - 1):
            current = 0
            for j in range(len(self.inputs)):
                current += (self.weights_hidden_input[i][j] * self.inputs[j])
            self.hidden[i] = sigmoid(current)

        for i in range(len(self.outputs)):
            current = 0
            for j in range(len(self.hidden)):
                current += (self.weights_output_hidden[i][j] * self.hidden[j])
            self.outputs[i] = sigmoid(current)

        # perform winner-takes-all for the network
        best = 0
        bestValue = self.outputs[0]
        for i in range(1, len(self.outputs)):
            if(self.outputs[i] > bestValue):
                best = i
                bestValue = self.outputs[i]
        return best

    def backpropagate(self, target, learning_rate):
        # given a classification, backpropagate the error through the weights.

        # calculate output node error
        err_out = [(target[out] - self.outputs[out]) * sigmoid_(self.outputs[out])
                   for out in range(len(self.outputs))]

        # calculate the hidden node error
        err_hid = zeros(len(self.hidden))
        for hid in range(len(self.hidden)):
            current = 0
            for out in range(len(self.outputs)):
                current += err_out[out] * self.weights_output_hidden[out][hid]
            err_hid[hid] = current * sigmoid_(self.hidden[hid])
        # adjust the hidden to output layer weights
        for out in range(len(self.outputs)):
            for hid in range(len(self.hidden)):
                self.weights_output_hidden[out][hid] += learning_rate * \
                    err_out[out] * self.hidden[hid]
        # adjust the input to hidden layer weights
        for hid in range(len(self.hidden) - 1):
            for inp in range(len(self.inputs)):
                self.weights_hidden_input[hid][inp] += learning_rate * \
                    err_hid[hid] * self.inputs[inp]

    def train(self, max_iter=30000, training_rate=0.05):
        # train the network from the test vectors
        for i in range(max_iter):
            test = randrange(self.count)
            self.feed_forward(self.dataset[test])
            self.backpropagate(self.target[test], training_rate)

    def test(self, n_test=10):
        # test the network given random vectors
        for i in range(n_test):
            test = randrange(self.count)
            result = self.feed_forward(self.dataset[test])
            print("#%d %s classified as %s (%s)" %
                  (test, self.dataset[test], result, self.target[test]))

    def testAll(self):
        # test the network by all data
        correct = 0
        for i in range(self.count):
            result = self.feed_forward(self.dataset[i])
            print("#%d %s classified as %s (%s)" %
                  (i, self.dataset[i], result, self.target[i]))
            if(self.target[i][result] == 1):
                correct += 1
        print("Classification rate %f" % (correct / self.count))


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="file", type=str,
                        default="data/iris.csv", help="data file")
    parser.add_argument('-s', '--split', dest='split', default=0.67, type=float,
                        help="data split ratio")
    return parser.parse_args()


def main():
    args = get_args()
    dataset, target = load_dataset(args.file, binarize=True)
    nn = Backpropagation(dataset, target)
    nn.train()
    nn.test()
    # nn.testAll()


if __name__ == '__main__':
    main()
