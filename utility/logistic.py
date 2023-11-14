#     logistic.py belongs to pyML
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

from enum import Enum
import numpy as np

from .functions import sigmoid

class Methods(Enum):
    GD = 0
    SGD = 1


class LogisticRegression:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.weights = np.zeros(0)

    @staticmethod
    def log_loss(p, y):
        # log loss function
        # Note: you can also use https://scikit-[Log Loss from scikit-learn](learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
        # clip for overflow
        p = np.clip(p, 1e-15, 1 - 1e-15)
        # calculate log loss
        return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()

    @staticmethod
    def sse_loss(p, y):
        # sum of squared errors loss function
        return np.sum((p - y) ** 2)

    def next_batch(self, x, y, batch_size):
        # loop over our dataset `X` in mini-batches of size `batchSize`
        for i in np.arange(0, x.shape[0], batch_size):
            # yield a tuple of the current batched data and labels
            yield (x[i : i + batch_size], y[i : i + batch_size])

    def fit(
        self,
        x,
        y,
        method=Methods.GD,
        lr=0.05,
        epochs=10000,
        batch_size=100,
        tolerance=1e-06,
    ):
        return (
            self.gd(x, y, lr, epochs)
            if method is Methods.GD
            else self.sgd(x, y, lr, epochs, batch_size, tolerance)
        )

    def sgd(self, x, y, lr, epochs, batch_size, tolerance):
        # weights initialization
        self.weights = np.zeros(x.shape[1])
        loss_values = []

        for i in range(epochs):
            epoch_loss = []
            for bx, by in self.next_batch(x, y, batch_size):
                # calculate sigmoid
                yp = self.calculate(bx)

                # calculate the error
                error = yp - by

                # calculate the gradient
                gradient = np.dot(bx.T, error) / by.size

                # update weights
                self.weights -= lr * gradient

                # calculate new sigmoid
                yp = self.calculate(bx)

                # calculate the loss
                loss = self.sse_loss(yp, by)

                # display loss
                if self.verbose and i % 1000 == 0:
                    print(f"loss in iteration {i} -> {loss} \t")

                # collect loss values
                epoch_loss.append(loss)
            loss_values.append(np.average(epoch_loss))
        # return loss values
        return loss_values

    def gd(self, x, y, lr, epochs=10000):
        # weights initialization
        self.weights = np.zeros(x.shape[1])
        loss_values = []

        for i in range(epochs):
            # calculate sigmoid using currrent weights
            yp = self.calculate(x)

            # calculate the error
            error = yp - y

            # calculate the gradient
            gradient = np.dot(x.T, error) / y.size

            # update weights
            self.weights -= lr * gradient

            # calculate sigmoid using new weights
            yp = self.calculate(x)

            # calculate the loss
            loss = self.log_loss(yp, y)

            # display loss
            if self.verbose and i % 1000 == 0:
                print(f"loss in iteration {i} -> {loss} \t")

            # collect loss values
            loss_values.append(loss)
        # return loss values
        return loss_values

    def calculate(self, x):
        # Calculate x * W
        z = np.dot(x, self.weights)
        # Calculate the sigmoid
        return sigmoid(z)

    def predict(self, x):
        # Predict using sigmoid calculation and return binary result
        return self.calculate(x).round()