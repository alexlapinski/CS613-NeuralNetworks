import math
import numpy as np


def sigmoid(x):
    return 1.0 / (1 + math.exp(-1*x))


class Perceptron:

    @staticmethod
    def from_weights(weights):
        return Perceptron(len(weights), weights=weights)

    def __init__(self, num_inputs, threshold=None,
                 activation_function=sigmoid, weights=None,
                 learning_rate=1.0):
        """
        Create a new perceptron
        :param num_inputs: The number of inputs this perceptron will handle
        :param threshold: The threshold to apply to the output of activation, if not specified,
                          the output of the activation function is returned
        :param activation_function: The activation function to apply to the output
        :param weights: A numpy array of pre-computed weights
        :param learning_rate: The learning rate for back-propagation
        """
        np.random.seed(0)

        self._prior_inputs = None
        self._prior_output = None
        self._weights = np.random.uniform(-1, 1, num_inputs) if weights is None else weights
        self._threshold = threshold
        self._learning_rate = learning_rate
        self._activation_function = activation_function

    def __transfer(self, inputs):
        return inputs.dot(self._weights)

    @property
    def weights(self):
        return self._weights

    @property
    def threshold(self):
        return self._threshold

    def update(self, expected_output, is_output_node=False):
        """
        Update the weights for this perceptron using the given expected value
        :param expected_output: Expected value this perceptron should have produced
        :param is_output_node: True if this node is an output node
        :return: delta produced during the update
        """

        if is_output_node:
            error = expected_output - self._prior_output
            delta = error * self._prior_output * (1 - self._prior_output)
        else:
            delta = 0  # TODO

        self._weights += self._learning_rate * (delta * self._prior_inputs)

        return delta

    def evaluate(self, inputs):
        """
        Returns the final output of the perceptron given the inputs.
        :param inputs: An array of input values to evaluate
                       against this perceptron
        :returns perceptron output:
        """
        self._prior_inputs = inputs.copy()
        net_input = self.__transfer(inputs)
        output = self._activation_function(net_input)
        self._prior_output = output

        if self._threshold is None:
            return output
        else:
            return output > self._threshold
