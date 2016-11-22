from perceptron import Perceptron, sigmoid
import numpy as np


class ArtificialNeuralNetwork:

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

    def __init__(self, num_inputs, num_hidden_nodes, num_output_nodes, learning_rate=1.0):
        #self._hidden_nodes = self._create_nodes(num_inputs + 1, num_hidden_nodes, learning_rate, sigmoid)
        self._learning_rate = learning_rate
        self._hidden_weights = np.random.uniform(-1, 1, num_inputs+1)

        # Captures inputs from training cycle
        self._prior_inputs = None

        # Captures outputs of hidden layer
        self._prior_output_inputs = None

        self._output_nodes = self._create_nodes(num_hidden_nodes, num_output_nodes, learning_rate, sigmoid)

    @property
    def hidden_weights(self):
        return self._hidden_weights

    @property
    def output_nodes(self):
        return self._output_nodes

    def train(self, training_inputs, expected_outputs, max_iterations=1000):

        iteration = 0
        while iteration < max_iterations:
            self.evaluate(training_inputs)
            self.update(expected_outputs)
            iteration += 1

    def evaluate(self, inputs):

        # add bias input value
        inputs = self._insert_bias(inputs)
        self._prior_inputs = inputs.copy()

        #output_inputs = []
        #for hidden_node in self._hidden_nodes:
        #    output_inputs.append(hidden_node.evaluate(inputs).reshape(-1, 1))
        output_inputs = inputs.dot(self._hidden_weights)
        self._prior_output_inputs = output_inputs.copy()

        # Add output input value
        #output_inputs = np.hstack(output_inputs)

        final_outputs = []
        for output_node in self._output_nodes:
            final_outputs.append(output_node.evaluate(output_inputs).reshape(-1, 1))

        return np.array(final_outputs).reshape(-1, 1)

    def update(self, expected_outputs):

        # Update output node(s)
        output_node_deltas = []
        for i in xrange(len(self._output_nodes)):
            expected_output = expected_outputs[i]
            node = self._output_nodes[i]
            error = expected_output - node.prior_output
            delta = error * node.prior_output * (1 - node.prior_output)
            node.update(delta)
            output_node_deltas.append(delta)

        # Update Hidden Layers
        hidden_node_deltas = []
        for i in xrange(self._hidden_weights.shape[0]-1):
            # For each node, capture previous output
            #hidden_node = self._hidden_nodes[i]
            delta = 0
            for j in xrange(len(self._output_nodes)):
                output_node = self._output_nodes[j]
                output_delta = output_node_deltas[j]
                new_weight = output_node.weights[i]
                delta += new_weight * output_delta

            prior_output = self._prior_output_inputs[i]
            delta *= prior_output * (1 - prior_output)
            hidden_node_deltas.append(delta)

            #
            # for i in xrange(len(delta)):
            #   delta_i = delta[i]
            #   prior_input_i = self._prior_inputs[i]
            #   self._weights += self._learning_rate * (delta_i * prior_input_i)
            #
            self._hidden_weights += self._learning_rate * (delta.dot(self._prior_inputs))
            #hidden_node.update(delta)

