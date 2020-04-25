import unittest
import numpy as np
from model.logistic_regression import LogisticRegression


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.model = LogisticRegression()
        self.model.params["w"] = np.array([[1.], [2.]])
        self.model.params["b"] = 2.
        self.x, self.y = np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])

    def test_sigmoid_func(self):
        sigmoid_arr = self.model.sigmoid(np.array([0, 2]))
        expected_arr = np.array([0.5, 0.8807970779778823])
        self.assertSequenceEqual(expected_arr.tolist(), sigmoid_arr.tolist())

    def test_forward_propagation(self):
        self.model.forward_propagation(self.x, self.y)

        expected_dw = np.array([[0.998456014637956], [2.3950723884862066]])
        expected_db = 0.001455578136784208

        self.assertSequenceEqual(expected_dw.tolist(), self.model.grads["dw"].tolist())
        self.assertEqual(expected_db, self.model.grads["db"], )
        self.assertEqual(5.801545319394553, self.model.cost)

    def test_gradient_decent(self):
        self.model.gradient_decent(self.x, self.y, num_iterations=100, learning_rate=0.009, print_cost=False)

        expected_w = np.array([[0.19033590888604332], [0.12259159246561413]])
        expected_b = 1.9253598300845747
        expected_dw = np.array([[0.6775204222153582], [1.416254952638088]])
        expected_db = 0.21919450454067657

        self.assertSequenceEqual(expected_w.tolist(), self.model.params["w"].tolist())
        self.assertEqual(expected_b, self.model.params["b"])
        self.assertSequenceEqual(expected_dw.tolist(), self.model.grads["dw"].tolist())
        self.assertEqual(expected_db, self.model.grads["db"])

    def test_predict(self):
        self.model.params["w"] = np.array([[0.1124579], [0.23106775]])
        self.model.params["b"] = -0.3
        X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
        expected_predictions = np.array([[1., 1., 0.]])
        predictions = self.model.predict(X)

        self.assertSequenceEqual(expected_predictions.tolist(), predictions.tolist())


if __name__ == '__main__':
    unittest.main()
