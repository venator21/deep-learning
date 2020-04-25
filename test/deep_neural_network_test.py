import unittest
import numpy as np

from model.deep_neural_network import DeepNeuralNetwork


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize_parameters(self):
        # set seed for reliable results
        np.random.seed(3)
        # build model with 2 hidden layers and 1 output layer, weights and bias initialized upon model creation
        self.model = DeepNeuralNetwork([5, 4, 3])
        # expected values for weights of 1 hidden layer with seed 3
        excpected_W1 = np.array([[0.017886284734303187, 0.004365098505119894, 0.0009649746807200863,
                                  -0.01863492703364491, -0.0027738820251439907],
                                 [-0.0035475897926898676, -0.0008274148148245977, -0.0062700067682384735,
                                  -0.0004381816897592824, -0.004772180303595027],
                                 [-0.013138647533626821, 0.008846223804995846, 0.008813180422075299,
                                  0.017095730636529485, 0.0005003364217686021],
                                 [-0.0040467741460089085, -0.0054535994761953045, -0.015464773155829684,
                                  0.009823674342581601, -0.011010676301114757]])

        self.assertSequenceEqual(excpected_W1.tolist(), self.model.parameters["W1"].tolist())

    def test_linear_forward(self):
        # set seed for reliable results
        np.random.seed(1)
        # set activations (input X corresponds to A0) , weights and bias with random values based on seed 1
        A = np.random.randn(3, 2)
        W = np.random.randn(1, 3)
        b = np.random.randn(1, 1)
        # perform linear step
        Z, linear_cache = DeepNeuralNetwork.linear_forward(A, W, b)
        # excepted value for linear component Z
        expected_Z = np.array([[3.262953374654174, -1.2342998686128757]])

        self.assertSequenceEqual(expected_Z.tolist(), Z.tolist())

    def test_linear_activation_forward(self):
        # set seed for reliable results
        np.random.seed(2)
        # set activations , weights and bias with random values based on seed 2
        A_prev = np.random.randn(3, 2)
        W = np.random.randn(1, 3)
        b = np.random.randn(1, 1)
        # perform linear step followed by activation step with sigmoid and relu
        sigmoid_A, sigmoid_cache = DeepNeuralNetwork.linear_activation_forward(A_prev, W, b, activation="sigmoid")
        relu_A, relu_cache = DeepNeuralNetwork.linear_activation_forward(A_prev, W, b, activation="relu")
        # excepted value for activations with sigmoid and relu
        expected_sigmoid_A = np.array([[0.968900232783661, 0.11013289483978989]])
        expected_relu_A = np.array([[3.4389613134945427, 0.]])

        self.assertSequenceEqual(expected_sigmoid_A.tolist(), sigmoid_A.tolist())
        self.assertSequenceEqual(expected_relu_A.tolist(), relu_A.tolist())

    def test_forward_propagation(self):
        # set seed for reliable results
        np.random.seed(6)
        # set model with dummy input
        self.model = DeepNeuralNetwork([0])
        # set X, weights and bias with random values based on seed 6
        X = np.random.randn(5, 4)
        W1 = np.random.randn(4, 5)
        b1 = np.random.randn(4, 1)
        W2 = np.random.randn(3, 4)
        b2 = np.random.randn(3, 1)
        W3 = np.random.randn(1, 3)
        b3 = np.random.randn(1, 1)
        # set model parameters
        self.model.parameters = {"W1": W1,
                                 "b1": b1,
                                 "W2": W2,
                                 "b2": b2,
                                 "W3": W3,
                                 "b3": b3}
        # perform forward propagation
        AL, caches = self.model.forward_propagation(X)
        # excepted value for last activations
        excpected_AL = np.array([[0.03921667922250356, 0.7049892075012225, 0.19734387234713996, 0.04728177161977361]])

        self.assertSequenceEqual(excpected_AL.tolist(), AL.tolist())
        self.assertEqual(3, len(caches))

    def test_compute_cost(self):
        # set AL vector of probabilities and Y vector of labels
        AL = np.array([[.8, .9, 0.4]])
        Y = np.array([[1, 1, 0]])

        # calculate cross entropy cost
        cost = DeepNeuralNetwork.compute_cost(AL, Y)
        expected_cost = 0.2797765635793422

        self.assertEqual(expected_cost, cost)

    def test_linear_backward(self):
        # set seed for reliable results
        np.random.seed(1)
        # set dZ, activations, weights and bias with random values based on seed 1
        dZ = np.random.randn(3, 4)
        A = np.random.randn(5, 4)
        W = np.random.randn(3, 5)
        b = np.random.randn(3, 1)
        linear_cache = (A, W, b)
        # calculate gradients
        dA_prev, dW, db = DeepNeuralNetwork.linear_backward(dZ, linear_cache)
        # excepted value for gradients in respect to activations
        expected_dA_prev = np.array(
            [[-1.151713362800674, 0.06718465115487987, - 0.32046959512402307, 2.0981271157690573],
             [0.6034587881597019, -3.725087007081199, 5.817007414837718, -3.8432683612931164],
             [-0.4319552024857888, -1.309874173571225, 1.7235470475890922, 0.05070577635898947],
             [-0.38981414839982087, 0.6081124403750794, -1.259384235864949, 1.4719159268171638],
             [-2.5221492584556406, 2.678825515032264, -0.6794746486966324, 1.4811954761548511]])

        self.assertSequenceEqual(expected_dA_prev.tolist(), dA_prev.tolist())

    def test_backward_propagation(self):
        # set seed for reliable results
        np.random.seed(3)
        # set gradients, linear components and activations with random values based on seed 3
        AL = np.random.randn(1, 2)
        Y = np.array([[1, 0]])

        A1 = np.random.randn(4, 2)
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(3, 1)
        Z1 = np.random.randn(3, 2)
        linear_cache_activation_1 = ((A1, W1, b1), Z1)

        A2 = np.random.randn(3, 2)
        W2 = np.random.randn(1, 3)
        b2 = np.random.randn(1, 1)
        Z2 = np.random.randn(1, 2)
        linear_cache_activation_2 = ((A2, W2, b2), Z2)

        caches = (linear_cache_activation_1, linear_cache_activation_2)

        # calculate gradients
        grads = DeepNeuralNetwork.backward_propagation(AL, Y, caches)

        # excepted value for gradients W1, b1, dA1
        expected_dW1 = np.array([[0.4101000205028755, 0.07807203345047609, 0.13798443640617922, 0.10502167445111507],
                                 [0., 0., 0., 0.],
                                 [0.052836517520190085, 0.01005865436969424, 0.0177776560038907, 0.01353079557355724]])
        expected_db1 = np.array([[-0.22007063390291245], [0.], [-0.028353487740947435]])
        expected_dA1 = np.array([[0.1291316175545915, - 0.4401412678058249],
                                 [-0.1417565465329586, 0.48317296176062136],
                                 [0.016637075426056504, - 0.05670697548189487]])

        self.assertSequenceEqual(expected_dW1.tolist(), grads["dW1"].tolist())
        self.assertSequenceEqual(expected_db1.tolist(), grads["db1"].tolist())
        self.assertSequenceEqual(expected_dA1.tolist(), grads["dA1"].tolist())

    def test_gradient_decent(self):
        # set seed for reliable results
        np.random.seed(2)
        # set model with dummy input
        self.model = DeepNeuralNetwork([0])
        # set gradients, linear components and activations with random values based on seed 2
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(3, 1)
        W2 = np.random.randn(1, 3)
        b2 = np.random.randn(1, 1)
        self.model.parameters = {"W1": W1,
                                 "b1": b1,
                                 "W2": W2,
                                 "b2": b2}
        # set seed for reliable results
        np.random.seed(3)
        # set gradients, linear components and activations with random values based on seed 3
        dW1 = np.random.randn(3, 4)
        db1 = np.random.randn(3, 1)
        dW2 = np.random.randn(1, 3)
        db2 = np.random.randn(1, 1)
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        # perform gradient decent
        self.model.gradient_decent(grads, learning_rate=0.1)

        expected_W1 = np.array([[-0.5956206947485025, - 0.09991781227752841, - 2.145845842475655, 1.8266200787414377],
                                [-1.7656967649434232, - 0.8062714677293054, 0.5111555653062888, - 1.1825880189248468],
                                [-1.053570401964746, - 0.861285811890899, 0.6828405198826926, 2.203745774764999]])
        expected_b1 = np.array([[-0.046592411222349535], [-1.2888827514788117], [0.5340549563631029]])
        expected_W2 = np.array([[-0.5556919583463779, 0.03540549824080157, 1.3296489510585878]])
        expected_b2 = np.array([[-0.8461076927196785]])

        self.assertSequenceEqual(expected_W1.tolist(), self.model.parameters["W1"].tolist())
        self.assertSequenceEqual(expected_b1.tolist(), self.model.parameters["b1"].tolist())
        self.assertSequenceEqual(expected_W2.tolist(), self.model.parameters["W2"].tolist())
        self.assertSequenceEqual(expected_b2.tolist(), self.model.parameters["b2"].tolist())


if __name__ == '__main__':
    unittest.main()
