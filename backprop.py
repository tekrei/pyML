#!/usr/bin/python
# coding=utf-8
'''
Neural Network demonstration with Backpropagation learning and the Iris dataset
by M. Tim Jones
from https://github.com/mtimjones/backprop
Python code by tekrei
'''
from __future__ import division

import random
import math
import numpy

dataset = [
# Sepal Length, Sepal Width, Petal Length, Petal Width
# Iris-setosa
[ [ 5.1, 3.5, 1.4, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.9, 3.0, 1.4, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.7, 3.2, 1.3, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.6, 3.1, 1.5, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.0, 3.6, 1.4, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.4, 3.9, 1.7, 0.4 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.6, 3.4, 1.4, 0.3 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.0, 3.4, 1.5, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.4, 2.9, 1.4, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.9, 3.1, 1.5, 0.1 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.4, 3.7, 1.5, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.8, 3.4, 1.6, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.8, 3.0, 1.4, 0.1 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.3, 3.0, 1.1, 0.1 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.8, 4.0, 1.2, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.7, 4.4, 1.5, 0.4 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.4, 3.9, 1.3, 0.4 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.1, 3.5, 1.4, 0.3 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.7, 3.8, 1.7, 0.3 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.1, 3.8, 1.5, 0.3 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.4, 3.4, 1.7, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.1, 3.7, 1.5, 0.4 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.6, 3.6, 1.0, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.1, 3.3, 1.7, 0.5 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.8, 3.4, 1.9, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.0, 3.0, 1.6, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.0, 3.4, 1.6, 0.4 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.2, 3.5, 1.5, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.2, 3.4, 1.4, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.7, 3.2, 1.6, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.8, 3.1, 1.6, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.4, 3.4, 1.5, 0.4 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.2, 4.1, 1.5, 0.1 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.5, 4.2, 1.4, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.9, 3.1, 1.5, 0.1 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.0, 3.2, 1.2, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.5, 3.5, 1.3, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.9, 3.1, 1.5, 0.1 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.4, 3.0, 1.3, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.1, 3.4, 1.5, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.0, 3.5, 1.3, 0.3 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.5, 2.3, 1.3, 0.3 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.4, 3.2, 1.3, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.0, 3.5, 1.6, 0.6 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.1, 3.8, 1.9, 0.4 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.8, 3.0, 1.4, 0.3 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.1, 3.8, 1.6, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 4.6, 3.2, 1.4, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.3, 3.7, 1.5, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
[ [ 5.0, 3.3, 1.4, 0.2 ], [ 1.0, 0.0, 0.0 ] ],
# Iris-versicolor
[ [ 7.0, 3.2, 4.7, 1.4 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.4, 3.2, 4.5, 1.5 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.9, 3.1, 4.9, 1.5 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.5, 2.3, 4.0, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.5, 2.8, 4.6, 1.5 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.7, 2.8, 4.5, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.3, 3.3, 4.7, 1.6 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 4.9, 2.4, 3.3, 1.0 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.6, 2.9, 4.6, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.2, 2.7, 3.9, 1.4 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.0, 2.0, 3.5, 1.0 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.9, 3.0, 4.2, 1.5 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.0, 2.2, 4.0, 1.0 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.1, 2.9, 4.7, 1.4 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.6, 2.9, 3.6, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.7, 3.1, 4.4, 1.4 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.6, 3.0, 4.5, 1.5 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.8, 2.7, 4.1, 1.0 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.2, 2.2, 4.5, 1.5 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.6, 2.5, 3.9, 1.1 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.9, 3.2, 4.8, 1.8 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.1, 2.8, 4.0, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.3, 2.5, 4.9, 1.5 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.1, 2.8, 4.7, 1.2 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.4, 2.9, 4.3, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.6, 3.0, 4.4, 1.4 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.8, 2.8, 4.8, 1.4 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.7, 3.0, 5.0, 1.7 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.0, 2.9, 4.5, 1.5 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.7, 2.6, 3.5, 1.0 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.5, 2.4, 3.8, 1.1 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.5, 2.4, 3.7, 1.0 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.8, 2.7, 3.9, 1.2 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.0, 2.7, 5.1, 1.6 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.4, 3.0, 4.5, 1.5 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.0, 3.4, 4.5, 1.6 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.7, 3.1, 4.7, 1.5 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.3, 2.3, 4.4, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.6, 3.0, 4.1, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.5, 2.5, 4.0, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.5, 2.6, 4.4, 1.2 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.1, 3.0, 4.6, 1.4 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.8, 2.6, 4.0, 1.2 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.0, 2.3, 3.3, 1.0 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.6, 2.7, 4.2, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.7, 3.0, 4.2, 1.2 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.7, 2.9, 4.2, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 6.2, 2.9, 4.3, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.1, 2.5, 3.0, 1.1 ], [ 0.0, 1.0, 0.0 ] ],
[ [ 5.7, 2.8, 4.1, 1.3 ], [ 0.0, 1.0, 0.0 ] ],
# iris-virginica
[ [ 6.3, 3.3, 6.0, 2.5 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 5.8, 2.7, 5.1, 1.9 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.1, 3.0, 5.9, 2.1 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.3, 2.9, 5.6, 1.8 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.5, 3.0, 5.8, 2.2 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.6, 3.0, 6.6, 2.1 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 4.9, 2.5, 4.5, 1.7 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.3, 2.9, 6.3, 1.8 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.7, 2.5, 5.8, 1.8 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.2, 3.6, 6.1, 2.5 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.5, 3.2, 5.1, 2.0 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.4, 2.7, 5.3, 1.9 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.8, 3.0, 5.5, 2.1 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 5.7, 2.5, 5.0, 2.0 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 5.8, 2.8, 5.1, 2.4 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.4, 3.2, 5.3, 2.3 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.5, 3.0, 5.5, 1.8 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.7, 3.8, 6.7, 2.2 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.7, 2.6, 6.9, 2.3 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.0, 2.2, 5.0, 1.5 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.9, 3.2, 5.7, 2.3 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 5.6, 2.8, 4.9, 2.0 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.7, 2.8, 6.7, 2.0 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.3, 2.7, 4.9, 1.8 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.7, 3.3, 5.7, 2.1 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.2, 3.2, 6.0, 1.8 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.2, 2.8, 4.8, 1.8 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.1, 3.0, 4.9, 1.8 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.4, 2.8, 5.6, 2.1 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.2, 3.0, 5.8, 1.6 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.4, 2.8, 6.1, 1.9 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.9, 3.8, 6.4, 2.0 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.4, 2.8, 5.6, 2.2 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.3, 2.8, 5.1, 1.5 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.1, 2.6, 5.6, 1.4 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 7.7, 3.0, 6.1, 2.3 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.3, 3.4, 5.6, 2.4 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.4, 3.1, 5.5, 1.8 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.0, 3.0, 4.8, 1.8 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.9, 3.1, 5.4, 2.1 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.7, 3.1, 5.6, 2.4 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.9, 3.1, 5.1, 2.3 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 5.8, 2.7, 5.1, 1.9 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.8, 3.2, 5.9, 2.3 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.7, 3.3, 5.7, 2.5 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.7, 3.0, 5.2, 2.3 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.3, 2.5, 5.0, 1.9 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.5, 3.0, 5.2, 2.0 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 6.2, 3.4, 5.4, 2.3 ], [ 0.0, 0.0, 1.0 ] ],
[ [ 5.9, 3.0, 5.1, 1.8 ], [ 0.0, 0.0, 1.0 ] ]
]

def sigmoid(x): return ( 1.0 / ( 1.0 + numpy.exp( -x ) ) )

def sigmoid_d( x ): return  ( x * ( 1.0 - x ) )

class Backpropagation:
    
    def __init__(self, n_input=4, n_hidden=25, n_output=3):
        # Neuron cell values
        self.inputs = numpy.ones(n_input+1)
        self.hidden = numpy.ones(n_hidden+1)
        self.outputs = numpy.zeros(n_output)
        # initialize the network with random weights
        # weights are randomly chosen between 0-0.5
        self.weights_hidden_input = numpy.random.rand(n_hidden, n_input+1)/2
        self.weights_output_hidden = numpy.random.rand(n_output, n_hidden+1)/2

    def feed_forward(self, data):
        # given the test input, feed forward to the output
        self.inputs = data
        # calculate hidden layer outputs
        for i in range(len(self.hidden)-1):
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
        err_out = [(target[out] - self.outputs[out]) * sigmoid_d(self.outputs[out]) for out in range(len(self.outputs))]

        # calculate the hidden node error
        err_hid = numpy.zeros(len(self.hidden))
        for hid in range(len(self.hidden)):
            current = 0
            for out in range(len(self.outputs)):
                current += err_out[ out ] * self.weights_output_hidden[ out ][ hid ]
            err_hid[hid] = current * sigmoid_d(self.hidden[hid])
        # adjust the hidden to output layer weights
        for out in range(len(self.outputs)):
            for hid in range(len(self.hidden)):
                self.weights_output_hidden[out][hid] += learning_rate * err_out[out] * self.hidden[hid]
        # adjust the input to hidden layer weights
        for hid in range(len(self.hidden)-1):
            for inp in range(len(self.inputs)):
                self.weights_hidden_input[hid][inp] += learning_rate * err_hid[hid] * self.inputs[inp]

    def train(self, max_iter=30000, training_rate=0.05):
        # train the network from the test vectors
        for i in range(max_iter):
            test = random.randrange(len(dataset))
            self.feed_forward(dataset[test][0])
            self.backpropagate(dataset[test][1], training_rate)

    def test(self, n_test=10):
        # test the network given random vectors
        for i in range(n_test):
            test = random.randrange(len(dataset))
            result = self.feed_forward(dataset[test][0]);
            print("#{} {} classified as {} ({})").format(test, dataset[test][0], result, dataset[test][1]);
            
    def testAll(self):
        # test the network by all data
        correct = 0
        for i in range(len(dataset)):
            result = self.feed_forward(dataset[i][0]);
            print("#{} {} classified as {} ({})").format(i, dataset[i][0], result, dataset[i][1]);
            if(dataset[i][1][result]==1):
                correct +=1
        print("Classification rate {}").format(correct/len(dataset))

def main():
    nn = Backpropagation()
    nn.train()
    nn.test()
    #nn.testAll()
main()
