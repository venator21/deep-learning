from model.logistic_regression import LogisticRegression
from model.deep_neural_network import DeepNeuralNetwork
from keras.datasets import mnist


def main():
    # Load mnist data set
    (x_train_orig, y_train), (x_test_orig, y_test) = mnist.load_data()

    # Preprocess data
    x_train_flatten = x_train_orig.reshape(x_train_orig.shape[0], -1).T
    x_test_flatten = x_test_orig.reshape(x_test_orig.shape[0], -1).T

    x_train = x_train_flatten / 255
    x_test = x_test_flatten / 255

    for i in range(y_train.shape[0]):
        if y_train[i] == 5:
            y_train[i] = 1
        else:
            y_train[i] = 0

    for i in range(y_test.shape[0]):
        if y_test[i] == 5:
            y_test[i] = 1
        else:
            y_test[i] = 0

    y_train.resize(1, y_train.shape[0])

    # Logistic Regression example
    model = LogisticRegression()
    model.fit(x_train, y_train, x_test, y_test, num_iterations=500)

    # Logistic Regression example
    model = DeepNeuralNetwork([784, 20, 5, 1])
    model.fit(x_train, y_train, num_iterations=500)
    model.predict(x_test, y_test)




if __name__ == '__main__':
    main()
