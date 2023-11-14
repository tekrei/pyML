#     svm.py belongs to pyML
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
from .functions import sigmoid

class SVM:
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.weights = np.zeros(0)
        
    @staticmethod
    def cost(x, y, w, re):
        # calculate hinge loss
        n = x.shape[0]
        distances = 1 - y * (np.dot(x, w))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = re * (np.sum(distances) / n)

        # calculate cost
        cost = 1 / 2 * np.dot(w, w) + hinge_loss
        return cost
    
    @staticmethod
    def cost_gradient(bx, by, w, re):
        distance = 1 - (by * np.dot(bx, w))
        dw = np.zeros(len(w))
        
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = w
            else:
                di = w - (re * by[ind] * bx[ind])
            dw += di
        
        dw = dw/len(by)
        return dw

    def next_batch(self, x, y, batch_size):
        # loop over our dataset `X` in mini-batches of size `batchSize`
        for i in np.arange(0, x.shape[0], batch_size):
            # yield a tuple of the current batched data and labels
            yield (x[i:i + batch_size], y[i:i + batch_size])

    def fit(self, x, y, lr=0.05, epochs=10000, batch_size=100, e=1e-06, re=10000):
        # weights initialization
        self.weights = np.zeros(x.shape[1])
        loss_values = []
        
        for i in range(epochs):
            epoch_loss = []
            for (bx, by) in self.next_batch(x, y, batch_size):
                # calculate the gradient
                gradient = self.cost_gradient(bx, by, self.weights, re)

                # update weights
                self.weights -= lr * gradient

                # calculate the loss
                loss = self.cost(bx, by, self.weights, re)
                
                # display loss
                if self.verbose and i % 1000 == 0:
                    print(f'loss in iteration {i} -> {loss} \t')

                # stoppage criterion
                if len(epoch_loss) > 0 and abs(epoch_loss[-1] - loss) < e:
                    return loss_values
                    
                # collect loss values
                epoch_loss.append(loss)
            loss_values.append(np.average(epoch_loss))

        return loss_values

    def calculate(self, x):
        # Calculate x * W
        z = np.dot(x, self.weights)
        # Calculate the sigmoid
        return sigmoid(z)

    def predict(self, x):
        # Predict using sigmoid calculation and return binary result
        return self.calculate(x).round()