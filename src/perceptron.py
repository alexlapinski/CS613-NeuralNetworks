import math
import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-1*x))


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
        return (inputs * self._weights.reshape(-1,1)).sum(axis=1)

    @property
    def weights(self):
        return self._weights

    @property
    def threshold(self):
        return self._threshold

    @property
    def prior_output(self):
        return self._prior_output

    def update(self, delta):
        """
        Update the weights for this perceptron using the given delta value
        :param delta: Delta value to use to update weights
        :return: delta produced during the update
        """
        for i in xrange(len(delta)):
            delta_i = delta[i]
            prior_input_i = self._prior_inputs[i]
            self._weights += self._learning_rate * (delta_i * prior_input_i)

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
