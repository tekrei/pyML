#!/usr/bin/python3
# coding=utf-8
'''
A simple implementation of the perceptron and perceptron learning
by M. Tim Jones from https://github.com/mtimjones/perceptron

Python code by tekrei
'''
from __future__ import division

import random


class Perceptron:

    def initialize(self, wsize):
        random.seed()
        # initialize the weights with random values
        self.weights = []
        for i in range(wsize):
            self.weights.append(random.random())

    def feedforward(self, inputs):
        totalSum = 0.0
        # calculate inputs * weights
        for i in range(len(inputs)):
            totalSum += self.weights[i] * inputs[i]
        # add in the bias
        totalSum += self.weights[i]
        # activation function (1 if value >= 1.0)
        if(totalSum >= 1.0):
            return 1
        return 0

    def train(self, test, learning_rate, max_iter):
        current = 0
        while True:
            iteration_error = 0.0
            print("Current iteration: %d" % current);
            for i in range(len(test)):
                desired_output = test[i][0] or test[i][1]
                output = self.feedforward(test[i])
                error = desired_output - output
                print("%s or %s = %s (%s)" %
                      (test[i][0], test[i][1], output, desired_output))
                self.weights[0] += learning_rate * (error * test[i][0])
                self.weights[1] += learning_rate * (error * test[i][1])
                self.weights[2] += learning_rate * error
                iteration_error += error * error
            print("Iteration error %f\n" % iteration_error)
            current += 1
            if((iteration_error <= 0) or (current > max_iter)):
                break


def main():
    # train the boolean OR set
    test = [[0, 0], [0, 1], [1, 0], [1, 1]]
    perceptron = Perceptron()
    # weight for two inputs and a bias
    perceptron.initialize(3)
    print("initialized weights %f %f bias %f" %
          (perceptron.weights[0], perceptron.weights[1], perceptron.weights[2]))
    # train the perceptron with test data, learning rate and maximum iteration count
    perceptron.train(test, 0.1, 10)
    print("final weights %f %f bias %f" %
          (perceptron.weights[0], perceptron.weights[1], perceptron.weights[2]))


main()
