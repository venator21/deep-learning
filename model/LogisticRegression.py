import numpy as np


class LogisticRegression:
    def __init__(self):
        self.cost = 0
        self.costs = list()
        self.params = {"w": np.array([]), "b": 0}
        self.grads = {"dw": np.array([]), "db": 0}

    @staticmethod
    def sigmoid(z):
        """
        Returns sigmoid of z.

        This method calculates value of sigmoid func
        (in particular logistic func) based on argument z.

        :param z: any numerical struct that supports arithmetic operations.
        :return: sigmoid of z.
        """
        return 1.0 / (1 + np.exp(-z))

    @staticmethod
    def zeros_init(dim):
        """
         Creates numpy array w of zeros shaped (dim,1) and
         scalar b equal 0.

        :param dim: row dimension for weights w
        :return: tuple of weights and bias
        """
        w = np.zeros((dim, 1))
        b = 0

        return w, b

    def forward_propagation(self, x, y):
        """
        Implementation of forward prop that calculates activation
        based on sigmoid func, cost based on logistic regression cost func
        and weights / bias gradients.

        :param x: numpy array of training data
        :param y: numpy array of labels
        """
        # calculate number of examples (columns of x)
        m = x.shape[1]

        # calculate activation and cost
        a = self.sigmoid(np.dot(self.params["w"].T, x) + self.params["b"])
        self.cost = (- 1 / m) * np.sum(y * np.log(a) + (1 - y) * (np.log(1 - a)))

        # calculate gradients for weights and bias
        self.grads["dw"] = (1 / m) * np.dot(x, (a - y).T)
        self.grads["db"] = (1 / m) * np.sum(a - y)

    def gradient_decent(self, x, y, num_iterations, learning_rate, print_cost):
        """
        This method runs gradient decent and updates weights and bias. Gradient decent runs for
        user defined number of iterations and uses user defined learning rate alfa.

        :param x: numpy array of data
        :param y: numpy array of label
        :param num_iterations: number of training iterations
        :param learning_rate: alfa rate for gradient decent
        :param print_cost: True to print the loss every 100 steps
        :return: params - dictionary containing the weights w and bias b;
                 grads - dictionary containing the gradients of the weights and bias;
                 costs - list of all the costs computed during the optimization;
        """

        self.costs = []

        for i in range(num_iterations):

            # calculate weights gradients and cost
            self.forward_propagation(x, y)

            # update weights
            self.params["w"] = self.params["w"] - learning_rate * self.grads["dw"]
            self.params["b"] = self.params["b"] - learning_rate * self.grads["db"]

            # Record the costs
            if i % 100 == 0:
                self.costs.append(self.cost)

            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, self.cost))

    def predict(self, x):
        """
        This method computes prediction on given data examples X.

        :param x: numpy array of data
        :return: numpy array of predictions (true/false)
        """

        m = x.shape[1]
        y_prediction = np.zeros((1, m))
        w = self.params["w"].reshape(x.shape[0], 1)
        b = self.params["b"]

        # Compute vector "A" predicting the probabilities
        a = self.sigmoid(np.dot(w.T, x) + b)

        for i in range(a.shape[1]):

            # Convert probabilities to actual predictions, threshold = 0.5
            if a[0][i] > 0.5:
                y_prediction[0][i] = 1
            else:
                y_prediction[0][i] = 0

        return y_prediction

    def fit(self, x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5, print_cost=True):
        """
        This method performs training for logistic regression model and
        computes accuracy of the model.

        :param x_train: training set represented by a numpy array
        :param y_train: training labels represented by a numpy array
        :param x_test: test set represented by a numpy array
        :param y_test: test set represented by a numpy array
        :param num_iterations: number of iterations to run gradient decent
        :param learning_rate: learning rate used in gradient decent
        :param print_cost: if true, prints cost every 100 iterations
        """

        self.params["w"], self.params["b"] = self.zeros_init(x_train.shape[0])

        # Gradient descent (≈ 1 line of code)
        self.gradient_decent(x_train, y_train, num_iterations, learning_rate, print_cost)

        # Predict test/train set examples (≈ 2 lines of code)
        y_prediction_test = self.predict(x_test)
        y_prediction_train = self.predict(x_train)

        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

        d = {"Y_prediction_test": y_prediction_test,
             "Y_prediction_train": y_prediction_train, }

        return d
