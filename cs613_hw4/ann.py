import numpy as np


def sigma(x):
    return 1.0 / (1.0 + np.exp(-1*x))


class ANN(object):
    """
    Single input Artificial Neural Network.
    This implementation accepts a single (sample) for forward propagation.
    """

    @staticmethod
    def _insert_bias(inputs):
        num_rows = inputs.shape[0]
        ones = np.ones((num_rows, 1))
        return np.c_[inputs, ones]

    def __init__(self, num_inputs, num_hidden_nodes, num_output_nodes, learning_rate=1.0):
        self._num_inputs = num_inputs
        self._num_hidden_nodes = num_hidden_nodes
        self._num_output_nodes = num_output_nodes
        self._learning_rate = learning_rate

        # Weights from Input data points to hidden nodes
        self._hidden_weights = np.random.uniform(-1, 1, (num_inputs + 1)*num_hidden_nodes).reshape(num_inputs + 1, num_hidden_nodes)

        # Weights from Hidden node outputs to output node(s)
        self._output_weights = np.random.uniform(-1, 1, (num_hidden_nodes*num_output_nodes)).reshape(num_hidden_nodes, num_output_nodes)

        self._activation_function = sigma

    @property
    def hidden_weights(self):
        return self._hidden_weights

    @hidden_weights.setter
    def hidden_weights(self, value):
        assert value.shape == self._hidden_weights.shape, "Shape of input 'value' must match"
        self._hidden_weights = value

    @property
    def output_weights(self):
        return self._output_weights

    @output_weights.setter
    def output_weights(self, value):
        assert value.shape == self.output_weights.shape, "Shape of input 'value' must match"
        self._output_weights = value

    @property
    def prior_inputs(self):
        return self._prior_inputs

    @property
    def prior_hidden_outputs(self):
        return self._prior_hidden_outputs

    @property
    def prior_outputs(self):
        return self._prior_hidden_outputs

    def __forward_propagate(self, inputs):
        """
        Forward Propagate the input values to output values
        :param inputs: row vector of inputs, each column represents one feature
        :return: row vector of outputs, each column represents one output node's output
        """

        prepared_inputs = self._insert_bias(inputs)
        self._prior_inputs = prepared_inputs.copy()

        hidden_outputs = self._activation_function(prepared_inputs.dot(self._hidden_weights))
        self._prior_hidden_outputs = hidden_outputs.copy()

        outputs = self._activation_function(hidden_outputs.dot(self._output_weights))
        self._prior_outputs = outputs.copy()

        return outputs

    def __backward_propagate(self, expected_outputs, actual_outputs):
        """
        Backward propagate the error based on expected outputs through the node weights
        :param expected_outputs: row vector of expected outputs,
                                 each column represents the expected output of one output node
        :param actual_outputs: row vector of actual outputs,
                               each column represents the actual output of one output node
        :return: nothing
        """

        num_samples = self._prior_inputs.shape[0]

        def compute_weight_offset(delta, prior_input):
            return (self._learning_rate / float(num_samples)) * delta.T.dot(prior_input)

        # Output deltas,
        output_deltas = []
        for i in xrange(self._num_output_nodes):
            y_i = expected_outputs[:, i]
            o_i = actual_outputs[:, i]
            # Compute Output Delta
            output_delta = (y_i - o_i) * o_i * (1 - o_i)
            output_deltas.append(output_delta)

            # Update Output Weights
            h = self._prior_hidden_outputs
            offset = compute_weight_offset(output_delta, h)
            self._output_weights[:, i] += offset

        # Compute Inner Delta
        for i in xrange(self._num_hidden_nodes):
            sum_weighted_deltas = 0
            for k in xrange(self._num_output_nodes):
                theta = self._output_weights[i, k]
                output_delta = output_deltas[k]
                sum_weighted_deltas += output_delta * theta

            prior_output = self._prior_hidden_outputs[:, i]
            hidden_delta = sum_weighted_deltas * prior_output * (1 - prior_output)
            offset = compute_weight_offset(hidden_delta, self._prior_inputs)
            self._hidden_weights[:, i] += offset

    def update(self, expected_outputs):
        return self.__backward_propagate(expected_outputs, self._prior_outputs)

    def evaluate(self, inputs, threshold=None):
        """
        Evaluate the trained artificial neural network on a single sample
        :param inputs: A single sample (row vector, each column is a feature)
        :param threshold: A floating point threshold, if output is above this, true is returned, otherwise, false
        :return: The output of the neural network (row vector, each column an output of each output node)
        """
        output = self.__forward_propagate(inputs)
        if threshold is None:
            return output
        else:
            return output > threshold

    def train(self, inputs, expected_outputs, iterations=1000, verbose=False):
        """
        Train the neural network using a single sample.
        :param inputs: The single sample used to train the network (row vector, each column is a feature)
        :param expected_outputs: The expected outputs (row vector, each column is for each output node's expected value)
        :param iterations: Total number of iterations to train the network
        :return: nothing
        """

        for iteration in xrange(iterations):
            actual_outputs = self.__forward_propagate(inputs)
            self.__backward_propagate(expected_outputs, actual_outputs)
            if verbose:
                print "Iteration {0}; Output = {1}".format(iteration, actual_outputs)
