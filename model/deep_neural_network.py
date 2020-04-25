"""
This is deep neural network model with L-1 layers with relu activation function
and with final L_th layer with sigmoid activation function for binary classification
"""

import numpy as np
import model.dnn_utils as utils


class DeepNeuralNetwork:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = dict()
        self.costs = list()
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize parameters w with np.random and b with np.zeros.
        """

        L = len(self.layer_dims)  # number of layers in the network

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

    @staticmethod
    def linear_forward(A, W, b):
        """
        Implements the linear part of a layer's forward propagation.

        :param A: activations from previous layer (or input data)
        :param W: numpy array of weights matrix
        :param b: numpy array of bias vector
        :return: tuple of Z (the input to the activation function) and
        linear cache (a python tuple containing A, W and b, stored for computing the backward pass)
        """

        Z = np.dot(W, A) + b
        linear_cache = (A, W, b)

        return Z, linear_cache

    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        """
        Implements the linear step followed by activation step of a layer's forward propagation.

        :param A_prev: activations from previous layer (or input data)
        :param W: numpy array of weights matrix
        :param b: numpy array of bias vector
        :param activation: activation function to be used in this layer
        :return: tuple of A (the activations) and cache (tuple containing "linear_cache" and "activation_cache",
                 stored for computing the backward pass)
        """

        Z, linear_cache = DeepNeuralNetwork.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, activation_cache = utils.sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = utils.relu(Z)

        cache = (linear_cache, activation_cache)
        return A, cache

    def forward_propagation(self, X):
        """
        Forward propagation that uses relu activation func for L-1 layers and sigmoid function for output layer.

        :param X: numpy array of data
        :return: tuple of AL (the activations of output layer) and caches (list of cache with each element
        containing "linear_cache" and "activation_cache", stored for computing the backward pass)
        """

        caches = []
        A = X
        L = len(self.parameters) // 2  # number of layers in the neural network

        # L-1 layers with relu func
        for l in range(1, L):
            A_prev = A
            A, cache = DeepNeuralNetwork.linear_activation_forward(A_prev, self.parameters["W" + str(l)],
                                                                   self.parameters["b" + str(l)],
                                                                   activation="relu")
            caches.append(cache)

        # last layer with sigmoid func
        AL, cache = DeepNeuralNetwork.linear_activation_forward(A, self.parameters["W" + str(L)],
                                                                self.parameters["b" + str(L)],
                                                                activation="sigmoid")
        caches.append(cache)

        return AL, caches

    @staticmethod
    def compute_cost(AL, Y):
        """
        Computes cross-entropy cost based on AL probability vector and Y label vector

        :param AL: numpy array of probability vector corresponding to label predictions
        :param Y: numpy array of label vector
        :return: cross-entropy cost
        """

        m = Y.shape[1]  # number of examples

        # Compute loss from AL and Y.
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        cost = np.squeeze(cost)  # this turns e.g. [[17]] into 17).

        return cost

    @staticmethod
    def linear_backward(dZ, linear_cache):
        """
        Linear step of backward propagation for a single layer (layer l).

        :param dZ: Gradient of the cost with respect to the linear output (of current layer l)
        :param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        :return: tuple of dA_prev, dW, db - gradients of the cost with respect to activation of previous layer,
        weights W of current layer l and bias b of current layer l
        """

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, linear_cache[0].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(linear_cache[1].T, dZ)

        return dA_prev, dW, db

    @staticmethod
    def linear_activation_backward(dA, cache, activation):
        """
        Implements the linear step followed by activation step of a layer's backward propagation

        :param dA: post-activation gradient for current layer l
        :param cache: tuple of values (linear_cache, activation_cache) stored for computing backward propagation
        :param activation: the activation to be used in current layer l, string: "sigmoid" or "relu"
        :return: tuple of dA_prev, dW, db - gradients of the cost with respect to activation of previous layer,
        weights W of current layer l and bias b of current layer l
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = utils.relu_backward(dA, activation_cache)
            dA_prev, dW, db = DeepNeuralNetwork.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = utils.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = DeepNeuralNetwork.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    @staticmethod
    def backward_propagation(AL, Y, caches):
        """
        Backward propagation that computes gradients for relu units of L-1 layers and sigmoid unit for output layer.
        :param AL: numpy array of probability vector, output of the forward propagation
        :param Y: numpy array of label vector
        :param caches: list of caches containing every cache of linear_activation_forward() with "relu"
        and the cache of linear_activation_forward() with "sigmoid"
        :return: dictionary with gradients
        """

        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # make sure Y is the same shape as AL

        # Initializing the backwards propagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL

        # Lth layer gradients (sigmoid)
        # Inputs: "dAL, current_cache"
        # Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], \
        grads["dW" + str(L)], \
        grads["db" + str(L)] = DeepNeuralNetwork.linear_backward(utils.sigmoid_backward(dAL, current_cache[1]),
                                                                 current_cache[0])

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer gradients (relu)
            # Inputs: "grads["dA" + str(l + 1)], current_cache"
            # Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            current_cache = caches[l]
            dA_prev_temp, \
            dW_temp, \
            db_temp = DeepNeuralNetwork.linear_backward(utils.relu_backward(grads["dA" + str(l + 1)], current_cache[1]),
                                                        current_cache[0])

            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def gradient_decent(self, grads, learning_rate):
        """
        Runs gradient decent algorithm on parameters of the model

        :param grads: dictionary containing gradients from backward propagation
        :param learning_rate: alfa rate for gradient decent
        """

        L = len(self.parameters) // 2  # number of layers in the neural network

        # Update parameters
        for l in range(L):
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - learning_rate * grads[
                "dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - learning_rate * grads[
                "db" + str(l + 1)]

    def fit(self, X, Y, num_iterations=500, learning_rate=0.0075, print_cost=True):
        """
        This method performs training for deep neural network and
        computes accuracy of the model.

        :param X: training set represented by a numpy array
        :param Y: training labels represented by a numpy array
        :param num_iterations: learning rate used in gradient decent
        :param learning_rate: number of iterations to run gradient decent
        :param print_cost: if true, prints cost every 100 iterations
        """

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation
            AL, caches = self.forward_propagation(X)

            # Compute cost.
            cost = DeepNeuralNetwork.compute_cost(AL, Y)

            # Backward propagation.
            grads = DeepNeuralNetwork.backward_propagation(AL, Y, caches)

            # Update parameters.
            self.gradient_decent(grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                self.costs.append(cost)

        print("Cost after iteration %i: %f" % (num_iterations, cost))
        self.costs.append(cost)

    def predict(self, X, Y):
        """
        This function is used to predict the results of a  L-layer neural network.

        :param X: data set of examples
        :param Y: data set of labels
        :return: predictions for the given dataset X
        """

        m = X.shape[1]
        p = np.zeros((1, m))

        # Forward propagation
        probabilities, caches = self.forward_propagation(X)

        # convert probabilities to 0/1 predictions
        for i in range(0, probabilities.shape[1]):
            if probabilities[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        print("Accuracy: " + str(np.sum((p == Y) / m)))

        return p
