from model.LogisticRegression import LogisticRegression
from keras.datasets import mnist


def main():
    # Logistic Regression example

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

    # train model
    model = LogisticRegression()
    model.fit(x_train, y_train, x_test, y_test, num_iterations=500)


if __name__ == '__main__':
    main()
