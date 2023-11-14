#     backpropagation.py belongs to pyML
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

import numpy as np
from random import randrange
from time import time

from .functions import sigmoid, sigmoid_prime

class Backpropagation:
    # Initialize the NN with n_hidden neurons
    def __init__(self, dataset, target, n_hidden=25):
        self.dataset = dataset
        self.count = len(self.dataset)
        n_input = len(self.dataset[0])
        self.target = target
        n_output = len(self.target[0])
        # Neuron cell values
        self.inputs = np.ones(n_input + 1)
        self.hidden = np.ones(n_hidden + 1)
        self.outputs = np.zeros(n_output)
        # initialize the network with random weights
        # weights are randomly chosen between 0-0.5
        self.weights_hidden_input = np.random.rand(n_hidden, n_input + 1) / 2
        self.weights_output_hidden = np.random.rand(n_output, n_hidden + 1) / 2

    def feed_forward(self, data):
        # given the test input, feed forward to the output
        self.inputs = data
        # calculate hidden layer outputs
        for i in range(len(self.hidden) - 1):
            current = 0
            for j in range(len(self.inputs)):
                current += self.weights_hidden_input[i][j] * self.inputs[j]
            self.hidden[i] = sigmoid(current)

        for i in range(len(self.outputs)):
            current = 0
            for j in range(len(self.hidden)):
                current += self.weights_output_hidden[i][j] * self.hidden[j]
            self.outputs[i] = sigmoid(current)

        # perform winner-takes-all for the network
        best = 0
        bestValue = self.outputs[0]
        for i in range(1, len(self.outputs)):
            if self.outputs[i] > bestValue:
                best = i
                bestValue = self.outputs[i]
        return best

    def backpropagate(self, target, learning_rate):
        # given a classification, backpropagate the error through the weights.
        # calculate output node error
        err_out = [
            (target[out] - self.outputs[out]) * sigmoid_prime(self.outputs[out])
            for out in range(len(self.outputs))
        ]

        # calculate the hidden node error
        err_hid = np.zeros(len(self.hidden))
        for hid in range(len(self.hidden)):
            current = 0
            for out in range(len(self.outputs)):
                current += err_out[out] * self.weights_output_hidden[out][hid]
            err_hid[hid] = current * sigmoid_prime(self.hidden[hid])
        # adjust the hidden to output layer weights
        for out in range(len(self.outputs)):
            for hid in range(len(self.hidden)):
                self.weights_output_hidden[out][hid] += (
                    learning_rate * err_out[out] * self.hidden[hid]
                )
        # adjust the input to hidden layer weights
        for hid in range(len(self.hidden) - 1):
            for inp in range(len(self.inputs)):
                self.weights_hidden_input[hid][inp] += (
                    learning_rate * err_hid[hid] * self.inputs[inp]
                )

    def train(self, max_iter=30000, training_rate=0.05):
        start = time()
        # train the network from the test vectors
        for i in range(max_iter):
            test = randrange(self.count)
            self.feed_forward(self.dataset[test])
            self.backpropagate(self.target[test], training_rate)
        print(f"Training is finished in {time()-start} seconds")

    def test(self, n_test=10):
        # test the network given randomly chosen vectors
        correct = 0
        for i in range(n_test):
            test = randrange(self.count)
            result = self.feed_forward(self.dataset[test])
            print(
                f"#{test} {self.dataset[test]} classified as {result} ({self.target[test]})"
            )
            if self.target[test][result] == 1:
                correct += 1
        print(f"Classification rate {correct / n_test}")

    def test_all(self):
        # test the network by all training data
        correct = 0
        for i in range(self.count):
            result = self.feed_forward(self.dataset[i])
            print(f"#{i} {self.dataset[i]} classified as {result} ({self.target[i]})")
            if self.target[i][result] == 1:
                correct += 1
        print(f"Classification rate {correct / self.count}")