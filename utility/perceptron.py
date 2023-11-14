#     perceptron.py belongs to pyML
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

from random import random

class Perceptron:
    def initialize(self, wsize):
        # initialize the weights with random values
        self.weights = []
        for i in range(wsize):
            self.weights.append(random())

    def feedforward(self, inputs):
        total_sum = 0.0
        # calculate inputs * weights
        for i in range(len(inputs)):
            total_sum += self.weights[i] * inputs[i]
        # add in the bias
        total_sum += self.weights[i]
        # activation function (1 if value >= 1.0)
        if total_sum >= 1.0:
            return 1
        return 0

    def train(self, test, learning_rate, max_iter):
        current = 0
        while True:
            iteration_error = 0.0
            print(f"Current iteration: {current}")
            for i in range(len(test)):
                desired_output = test[i][0] or test[i][1]
                output = self.feedforward(test[i])
                error = desired_output - output
                print(f"\t{test[i][0]} or {test[i][1]} = {output} ({desired_output})")
                self.weights[0] += learning_rate * (error * test[i][0])
                self.weights[1] += learning_rate * (error * test[i][1])
                self.weights[2] += learning_rate * error
                iteration_error += error * error
            print(f"Iteration error {iteration_error}")
            current += 1
            if (iteration_error <= 0) or (current > max_iter):
                break