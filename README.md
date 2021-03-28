# Machine Learning Implementations in Python

This repository contains different machine learning algorithm implementations, inspired from resources. It is
using [Jupyter notebooks](https://jupyter.org/) and [JupyterLab](http://jupyterlab.io/).

## Usage

Install dependencies

    poetry update

Start JupyterLab

    poetry run jupyter lab

or

    poetry shell
    jupyter lab

## Implemented Algorithms

- Comparison of various classification models
  from [scikit-learn](http://scikit-learn.org/): [model_comparison.ipynb](./model_comparison.ipynb)
- Optimization and root finding: [optimization.ipynb](./optimization.ipynb)
  - [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
  - [Bisection Method](https://en.wikipedia.org/wiki/Bisection_method)
  - [Secant Method](https://en.wikipedia.org/wiki/Secant_method)
  - [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method)
- [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression): [linear_regression.ipynb](./linear_regression.ipynb)
- [K-Means](https://en.wikipedia.org/wiki/K-means_clustering): [kmeans.ipynb](./kmeans.ipynb)
- [K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm):  [knn.ipynb](./knn.ipynb)
- [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier):  [naive_bayes.ipynb](./naive_bayes.ipynb)
- [Perceptron](https://en.wikipedia.org/wiki/Perceptron):  [perceptron.ipynb](./perceptron.ipynb)
- [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation):  [backpropagation.ipynb](./backpropagation.ipynb)
- [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree):  [decision_tree.ipynb](./decision_tree.ipynb)
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression):  [logistic_regression.ipynb](./logistic_regression.ipynb)
- [Feedforward Neural Network](https://en.wikipedia.org/wiki/Feedforward_neural_network):  [feedforward.ipynb](./feedforward.ipynb)
- [Support Vector Machine](https://en.wikipedia.org/wiki/Support-vector_machine): [svm.ipynb](./svm.ipynb)

### Links & Resources

- [scikit-learn machine learning library for Python](http://scikit-learn.org/)
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
- [The Elements of Statistical Learning](http://web.stanford.edu/~hastie/ElemStatLearn/)
- [Introduction to Computation and Programming Using Python](https://mitpress.mit.edu/books/introduction-computation-and-programming-using-python-1)
- [Algorithms From Scratch](http://machinelearningmastery.com/category/algorithms-from-scratch/)
- [Kaggle](https://www.kaggle.com/)
- [UCI Machine Learning Resources](https://archive.ics.uci.edu/ml/index.php)

## Package Management

This project is using [poetry](https://python-poetry.org/) Python package and dependency manager.

- Init interactively `poetry init`
- Add package `poetry add package-name`
- Remove package `poetry remove package-name`
- Install dependencies `poetry install`
- Update dependencies `poetry update`
- Show available packages `poetry show`
- Run a command in the virtualenv `poetry run command`
- Open virtualenv `poetry shell`
