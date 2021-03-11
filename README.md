# pyML: Machine Learning Implementations, Experiments in Python

Different machine learning algorithm implementations, inspired from various resources. We are using [Jupyter notebooks](https://jupyter.org/) and [JupyterLab](http://jupyterlab.io/).

## Usage

Install dependencies

   poetry update

Start JypterLab

   poetry run jupyter lab

or

   poetry shell
   jupyter lab

## Implemented Algorithms

- Optimization and root finding: [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent), [Bisection Method](https://en.wikipedia.org/wiki/Bisection_method), [Secant Method](https://en.wikipedia.org/wiki/Secant_method), [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method)
- [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [K-Means](https://en.wikipedia.org/wiki/K-means_clustering)
- [K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Perceptron](https://en.wikipedia.org/wiki/Perceptron)
- [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
- [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree)

### Links & Resources

- [scikit-learn machine learning library for Python](http://scikit-learn.org/)
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
- [The Elements of Statistical Learning](http://web.stanford.edu/~hastie/ElemStatLearn/)
- [Introduction to Computation and Programming Using Python](https://mitpress.mit.edu/books/introduction-computation-and-programming-using-python-1)
- [Algorithms From Scratch](http://machinelearningmastery.com/category/algorithms-from-scratch/)
- [Kaggle](https://www.kaggle.com/)
- [UCI Machine Learning Resources](https://archive.ics.uci.edu/ml/index.php)

## Package Management

We are using [poetry](https://python-poetry.org/) Python package and dependency manager.

- Init interactively `poetry init`
- Add package `poetry add package-name`
- Remove package `poetry remove package-name`
- Install dependencies `poetry install`
- Update dependencies `poetry update`
- Show available packages `poetry show`
- Run a command in the virtualenv `poetry run command`
- Open virtualenv `poetry shell`
