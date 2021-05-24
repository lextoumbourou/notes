"""
An attempt at building a neural network from scratch.

Based on learnings from Neural Networks and Deep Learning.
"""

import numpy as np


def sigmoid(X):
    """Sigmoid function."""
    return 1. / (1. + np.exp(-X))


def relu(X):
    """Max(0, x) for some array."""
    return np.maximum(0, X)


def initialize_params(n_x, n_h, n_y):
    """Init params."""
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    return dict(W1=W1, b1=b1, W2=W2, b2=b2)


def initialize_params_deeps(layer_dims):
    """Init params for entire network."""
    parameters = {}

    L = len(layer_dims)

    for l in range(1, L):
        parameters['W{l}'.format(l=l)] = np.random.randn(
            layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b{l}'.format(l=l)] = np.random.randn(layer_dims[l], 1) * 0.01

    return parameters


def linear_forward(A, W, b):
    """Linear forward pass."""
    Z = np.dot(W, A) + b

    assert Z.shape == (W.shape[0], A.shape[1])

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """Linear activation forward."""
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters['W{l}'.format(l=l)],
            parameters['b{l}'.format(l=l)], activation='relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(
            A, parameters['W{L}'.format(L=L)],
            parameters['b{L}'.format(L=L)], activation='sigmoid')
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = -(1./m) * np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1. - AL), 1. - Y))

    cost = np.squeeze(cost)

    assert cost.shape == ()

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = (1. / m) * np.dot(dZ, A_prev.T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # make Y the same shape as AL

    dAL = -(np.divide(Y, AL) - np.divide(1. - Y, 1. - AL))

    current_cache = caches[L - 1]
    grads['dA{L}'.format(L=L)], grads['dW{L}'.format(L=L)], grads['db{L}'.format(L=L)] = (
        linear_activation_backward(dAL, current_cache, activation='sigmoid'))

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads['dA{0}'.format(l + 2)], current_cache, activation='relu')
        grads['dA{0}'.format(l + 1)] = dA_prev_temp
        grads['dW{0}'.format(l + 1)] = dW_temp
        grads['db{0}'.format(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)

    for l in range(L):
        parameters['W{0}'.format(l + 1)] = parameters['W{0}'.format(l + 1)] - learning_rate * grads['dW{0}'.format(l + 1)]
        parameters['b{0}'.format(l + 1)] = parameters['b{0}'.format(l + 1)] - learning_rate * grads['db{0}'.format(l + 1)]

    return parameters
