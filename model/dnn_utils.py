"""
This modules holds utilities functions for deep neural network module.
"""

import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy.

    :param Z: linear output of layer
    :return: tuple of A - post-activation parameter and cache - linear output of layer
    for backward propagation calculations
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implements the relu activation in numpy.

    :param Z: linear output of layer
    :return: tuple of A - post-activation parameter and cache - linear output of layer
    for backward propagation calculations
    """

    A = np.maximum(0, Z)
    cache = Z

    return A, cache


def relu_backward(dA, cache):
    """
    Implements the backward propagation for a single relu unit.

    :param dA: post-activation gradient
    :param cache: 'Z' stored for computing backward propagation
    :return: Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # converting dz to a correct object of numpy array

    # When z <= 0, set dz to 0
    dZ[Z <= 0] = 0

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implements the backward propagation for a single sigmoid unit.

    :param dA: post-activation gradient
    :param cache: 'Z' stored for computing backward propagation
    :return: Gradient of the cost with respect to Z
    """

    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ