import numpy as np


def sigma(x):
    return 1.0 / (1.0 + np.exp(-1*x))


class SingleANN:
    """
    Single input Artificial Neural Network.
    This implementation accepts a single (sample) for forward propagation.
    """

    @staticmethod
    def _insert_bias(inputs):
        return np.r_[inputs, 1]

    def __init__(self, num_inputs, num_hidden_nodes, num_output_nodes, learning_rate=1.0, threshold=0.5):
        self._num_inputs = num_inputs
        self._num_hidden_nodes = num_hidden_nodes
        self._num_output_nodes = num_output_nodes
        self._learning_rate = learning_rate

        # Weights from Input data points to hidden nodes
        self._hidden_weights = [np.random.uniform(-1, 1, num_inputs + 1).reshape(-1, 1) for _ in xrange(num_hidden_nodes)]

        # Weights from Hidden node outputs to output node(s)
        self._output_weights = [np.random.uniform(-1, 1, num_hidden_nodes).reshape(-1, 1) for _ in xrange(num_output_nodes)]

        self._activation_function = sigma
        self._threshold = threshold

    @property
    def hidden_weights(self):
        return self._hidden_weights

    @property
    def output_weights(self):
        return self._output_weights

    def __forward_propagate(self, inputs, verbose=False):
        """
        Forward Propagate the input values to output values
        :param inputs: row vector of inputs, each column represents one feature
        :return: row vector of outputs, each column represents one output node's output
        """

        if verbose:
            print "Original Inputs", inputs
        prepared_inputs = self._insert_bias(inputs)

        if verbose:
            print "Prepared_inputs", prepared_inputs

        self._prior_inputs = prepared_inputs.copy()

        hidden_outputs = []
        for i in xrange(len(self._hidden_weights)):
            hidden_weights = self._hidden_weights[i]
            temp_hidden_outputs = prepared_inputs.dot(hidden_weights)
            hidden_outputs.append(self._activation_function(temp_hidden_outputs))

        hidden_outputs = np.array(hidden_outputs).flatten()
        if verbose:
            print "Hidden Outputs", hidden_outputs

        self._prior_hidden_outputs = hidden_outputs.copy()

        outputs = []
        for i in xrange(len(self._output_weights)):
            output_weights = self._output_weights[i]
            temp_outputs = hidden_outputs.dot(output_weights)
            outputs.append(self._activation_function(temp_outputs))

        outputs = np.array(outputs).flatten()
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

        def compute_weight_offset(delta, prior_input):
            return (self._learning_rate * delta * prior_input).reshape(-1, 1)

        output_deltas = []
        for i in xrange(len(self._output_weights)):
            y_i = expected_outputs[i]
            o_i = actual_outputs[i]
            # Compute Output Delta
            output_delta = (y_i - o_i) * o_i * (1 - o_i)
            output_deltas.append(output_delta)

            # Update Output Weights
            h = self._prior_hidden_outputs.flatten()
            offset = compute_weight_offset(output_delta, h)
            self._output_weights[i] += offset

        # Compute Inner Delta
        sum_weighted_deltas = 0
        for i in xrange(len(output_deltas)):
            sum_weighted_deltas = (output_deltas[i] * self._output_weights[i]).sum(axis=0)

        for i in xrange(len(self._hidden_weights)):
            h_i = self._prior_hidden_outputs[i]
            hidden_delta = sum_weighted_deltas * h_i * (1 - h_i)
            offset = compute_weight_offset(hidden_delta, self._prior_inputs)
            self._hidden_weights[i] += offset

    def evaluate(self, inputs, threshold=0.5, verbose=False):
        """
        Evaluate the trained artificial neural network on a single sample
        :param inputs: A single sample (row vector, each column is a feature)
        :return: The output of the neural network (row vector, each column an output of each output node)
        """
        return self.__forward_propagate(inputs, verbose) > threshold

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


class BatchANN:

    @staticmethod
    def _create_nodes(num_inputs, num_nodes, learning_rate, activation_function):
        nodes = []
        # Account for bias in inputs
        for i in xrange(num_nodes):
            nodes.append(Perceptron(num_inputs, learning_rate=learning_rate,
                                    activation_function=activation_function))

        return nodes

    @staticmethod
    def _insert_bias(inputs):
        return np.c_[inputs, np.ones(inputs.shape[0])]

    @staticmethod
    def compute_inner_delta(output_delta, output_weights):
        return (output_weights.dot(output_delta)).T

    @staticmethod
    def compute_output_delta(actual_outputs, expected_outputs):
        """
        Compute the delta for an output node given actual outputs and expected outputs.
        :param actual_outputs: Column matrix of actual outputs (1 per sample)
        :param expected_outputs: Column matrix of expected outputs (1 per sample)
        :return: Column vector of the same size as outputs
        """

        assert actual_outputs.shape == expected_outputs.shape, "Cannot compute delta of differing matrices"

        return (expected_outputs - actual_outputs) * expected_outputs * (1 - expected_outputs)

    @staticmethod
    def compute_weight_offset(learning_rate, delta, prior_inputs):
        """
        Compute the offset to apply to a given node's weight.
        :param learning_rate: Learning rate for the offset
        :param delta: Delta for the given node
        :param prior_inputs: prior inputs to the given node
        :return: offset for each weight of the node's inputs
        """
        num_samples = prior_inputs.shape[0] # Samples are stacked vertically
        return (learning_rate/float(num_samples) * delta.T).dot(prior_inputs)

    def __init__(self, num_inputs, num_hidden_nodes, learning_rate=1.0):
        self._learning_rate = learning_rate
        self._hidden_weights = np.random.uniform(-1, 1, num_inputs+1).reshape(-1, 1)
        self._output_weights = np.random.uniform(-1, 1, num_hidden_nodes).reshape(-1, 1)

        # Captures inputs from training cycle
        self._prior_inputs = None

        # Captures outputs of hidden layer
        self._prior_hidden_outputs = None

        # Captures outputs of output node
        self._prior_outputs = None

    @property
    def hidden_weights(self):
        return self._hidden_weights

    @property
    def output_weights(self):
        return self._output_weights

    def train(self, training_inputs, expected_outputs, max_iterations=1000):

        iteration = 0
        while iteration < max_iterations:
            self.forward_propagate(training_inputs)
            self.backward_propagate(expected_outputs)
            iteration += 1

    def forward_propagate(self, inputs):

        print "input shape", inputs.shape

        # add bias input value
        inputs = self._insert_bias(inputs)
        self._prior_inputs = inputs.copy()

        hidden_outputs = sigma(inputs.dot(self._hidden_weights))
        self._prior_hidden_outputs = hidden_outputs.copy()

        print hidden_outputs.shape

        final_outputs = sigma(hidden_outputs.dot(self.output_weights))
        self._prior_outputs = final_outputs
        return final_outputs

    def backward_propagate(self, expected_outputs):

        # Compute Output Delta
        output_delta = self.compute_output_delta(self._prior_outputs, expected_outputs)

        # Update Output Weights
        self._output_weights += self.compute_weight_offset(self._learning_rate, output_delta, self._prior_hidden_outputs)

        # Compute Hidden Delta
        hidden_delta = self.compute_inner_delta(output_delta, self._output_weights)

        # Update Hidden Weights
        self._hidden_weights += self.compute_weight_offset(self._learning_rate, hidden_delta, self._prior_inputs)

