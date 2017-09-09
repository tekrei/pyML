#!/usr/bin/python3
# coding=utf-8
import numpy
import scipy.optimize

from utility import *


def compute_error(b, m, points):
    '''
    Error function: Mean Squared Error (MSE)
    MSE = (1/N)(sum((yi-(mxi+b))^2))
    '''
    error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        error += (y - (m * x + b)) ** 2
    return error / float(len(points))


def step_gradient(b, m, points, l_rate):
    '''
    Uses partial derivatives of each parameter (m and b) of the error function
    l_rate variable controls how large of a step we take downhill during each
    iteration
    '''
    b_ = 0
    m_ = 0
    N = len(points)
    for i in range(0, N):
        x = points[i, 0]
        y = points[i, 1]
        b_ += -(2.0 / N) * (y - ((m * x) + b))
        m_ += -(2.0 / N) * x * (y - ((m * x) + b))
    b1 = b - (l_rate * b_)
    m1 = m - (l_rate * m_)
    return [b1, m1]


def test_gd():
    '''
    A gradient descent and linear regression example to solve y = mx + b equation
    using gradient descent, m is slope, b is y-intercept
    by Matt Nedrich
    Source: http://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/
    '''
    # read data
    points = load_numbers("data/data.csv")
    # initial y-intercept guess
    b0 = 0
    # initial slope guess
    m0 = 0
    # number of iterations to perform the GD
    n_iter = 1000
    for i in range(n_iter):
        # perform GD iterations
        b0, m0 = step_gradient(b0, m0, numpy.array(points), 0.0001)
    print("GD\ti=%d\tb=%f\tm=%f\te=%f\t(y=%f*x+%f)" %
          (n_iter, b0, m0, compute_error(b0, m0, points), m0, b0))


def manual_gd(f_, x_old=0, x_new=5, learningRate=0.1, precision=0.00001):
    """
    A simple gradient descent usage for function optimization
    """
    iteration = 0
    while abs(x_new - x_old) > precision:
        iteration += 1
        x_old = x_new
        x_new = x_old - (learningRate * f_(x_old))
    print("MGD\ti=%d\tf(x)=%f" % (iteration, x_new))


def bisect(f, x0, x1, e=1e-12):
    """
    Bisection method for root finding
    """
    x = (x0 + x1) / 2.0
    n = 0
    while (x1 - x0) / 2.0 > e:
        n += 1
        if f(x) == 0:
            return x
        elif f(x0) * f(x) < 0:
            x1 = x
        else:
            x0 = x
        x = (x0 + x1) / 2.0
    print("B\ti=%d\tx=%f\tf(x)=%f" % (n, x, f(x)))
    return x


def secant(f, x0, x1, e=1e-12):
    """
    Secant method for root finding
    """
    n = 0
    while True:
        n += 1
        x = x1 - f(x1) * ((x1 - x0) / (f(x1) - f(x0)))
        if abs(x - x1) < e:
            break
        x0 = x1
        x1 = x
    print("S\ti=%d\tx=%f\tf(x)=%f" % (n, x, f(x)))
    return x


def newton(f, f_, x0, e=1e-12):
    """
    Newton method for root finding
    """
    n = 0
    while True:
        n += 1
        x = x0 - (f(x0) / f_(x0))
        if abs(x - x0) < e:
            break
        x0 = x
    print("N\ti=%d\tx=%f\tf(x)=%f" % (n, x, f(x)))
    return x


def f(x): return x**2 - 4 * x + 1


def f_(x): return 2 * x - 4


def main():
    # linear regression with gradient descent
    test_gd()
    # function optimization with gradient descent
    manual_gd(f_)
    # root finding methods
    # bisection method
    bisect(f, 0, 2)
    # secant method
    secant(f, 1, 2)
    # newton method
    newton(f, f_, 1)
    # scipy
    print("SB\tx=%f" % scipy.optimize.bisect(f, 0, 2))
    print("SS\tx=%f" % scipy.optimize.newton(f, 1))
    print("SN\tx=%f" % scipy.optimize.newton(f, 1, f_))


if __name__ == "__main__":
    main()
