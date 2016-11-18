from perceptron import Perceptron, sigmoid
import numpy as np


class ArtificialNeuralNetwork:

    @staticmethod
    def _create_nodes(num_inputs, num_nodes, learning_rate, activation_function):
        nodes = []
        # Account for bias in inputs
        for i in xrange(num_nodes):
            nodes.append(Perceptron(num_inputs, learning_rate=learning_rate, activation_function=activation_function))

        return nodes

    @staticmethod
    def _insert_bias(inputs):
        return np.c_[inputs, np.ones(inputs.shape[0])]

    def __init__(self, num_inputs, num_input_nodes, num_hidden_nodes, num_output_nodes, learning_rate=1.0):
        self._input_nodes = self._create_nodes(num_inputs + 1, num_input_nodes, learning_rate, lambda x: x)
        self._hidden_nodes = self._create_nodes(num_input_nodes + 1, num_hidden_nodes, learning_rate, lambda x: x)
        self._output_nodes = self._create_nodes(num_hidden_nodes + 1, num_output_nodes, learning_rate, sigmoid)

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def hidden_nodes(self):
        return self._hidden_nodes

    @property
    def output_nodes(self):
        return self._output_nodes

    def evaluate(self, inputs):

        # add bias input value
        inputs = self._insert_bias(inputs)

        hidden_inputs = []
        for input_node in self._input_nodes:
            hidden_inputs.append(input_node.evaluate(inputs).reshape(-1, 1))

        # add bias hidden input value
        hidden_inputs = self._insert_bias(np.hstack(hidden_inputs))

        output_inputs = []
        for hidden_node in self._hidden_nodes:
            output_inputs.append(hidden_node.evaluate(hidden_inputs).reshape(-1,1))

        # Add output input value
        output_inputs = self._insert_bias(np.hstack(output_inputs))

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

        # Update Hidden Layers (if any)
        hidden_node_deltas = []
        if self._hidden_nodes is not None:
            for i in xrange(len(self.hidden_nodes)):
                # For each node, capture previous output
                hidden_node = self._hidden_nodes[i]
                delta = 0
                for j in xrange(len(self._output_nodes)):
                    output_node = self._output_nodes[j]
                    output_delta = output_node_deltas[j]
                    new_weight = output_node.weights[i]
                    delta += new_weight * output_delta

                prior_output = hidden_node.prior_output
                delta *= prior_output * (1 - prior_output)
                hidden_node_deltas.append(delta)
                hidden_node.update(delta)

        # Update Input Nodes
        for i in xrange(len(self._input_nodes)):
            # For each node, capture previous output
            input_node = self._input_nodes[i]
            delta = 0

            for j in xrange(len(self.hidden_nodes)):
                hidden_node = self._hidden_nodes[j]
                hidden_delta = hidden_node_deltas[j]
                new_weight = hidden_node.weights[i]
                delta += new_weight * hidden_delta

            prior_output = input_node.prior_output
            delta *= prior_output * (1 - prior_output)
            input_node.update(delta)

