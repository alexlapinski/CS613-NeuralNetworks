from perceptron import Perceptron
import numpy as np


class ArtificialNeuralNetwork:

    @staticmethod
    def _create_nodes(num_inputs, num_nodes, learning_rate):
        nodes = []
        # Account for bias in inputs
        num_inputs += 1
        for i in xrange(num_nodes):
            nodes.append(Perceptron(num_inputs,learning_rate=learning_rate))

        return nodes

    def __init__(self, num_inputs, num_input_nodes, num_hidden_nodes, num_output_nodes, learning_rate=1.0):
        self._input_nodes = self._create_nodes(num_inputs, num_input_nodes, learning_rate)

        if num_hidden_nodes == 0:
            self._hidden_nodes = None
            self._output_nodes = self._create_nodes(num_input_nodes, num_output_nodes, learning_rate)
        else:
            self._hidden_nodes = self._create_nodes(num_input_nodes, num_hidden_nodes, learning_rate)
            self._output_nodes = self._create_nodes(num_hidden_nodes, num_output_nodes, learning_rate)

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

        # add bias
        inputs = np.append(inputs, 1)

        hidden_inputs = []
        for input_node in self._input_nodes:
            hidden_inputs.append(input_node.evaluate(inputs))

        # add bias
        hidden_inputs.append(1)

        output_inputs = []
        if self._hidden_nodes is None:
            output_inputs = hidden_inputs
        else:
            for hidden_node in self._hidden_nodes:
                output_inputs.append(hidden_node.evaluate(np.array(hidden_inputs)))

        # Add Bias
        output_inputs.append(1)

        final_outputs = []
        for output_node in self._output_nodes:
            final_outputs.append(output_node.evaluate(np.array(output_inputs)))

        return final_outputs

    def update(self, expected_outputs):

        # Update output node(s)
        output_node_deltas = []
        for i in xrange(len(expected_outputs)):
            expected_output = expected_outputs[i]
            node = self._output_nodes[i]
            #print "Weights {0} before update: {1}".format(i, node.weights)
            error = expected_output - node.prior_output
            delta = error * node.prior_output * (1 - node.prior_output)
            node.update(delta)
            output_node_deltas.append(delta)
            #print "after update", delta, node.weights

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
            if self._hidden_nodes is None:
                for j in xrange(len(self._output_nodes)):
                    output_node = self._output_nodes[j]
                    output_delta = output_node_deltas[j]
                    new_weight = output_node.weights[i]
                    delta += new_weight * output_delta
            else:
                for j in xrange(len(self.hidden_nodes)):
                    hidden_node = self._hidden_nodes[j]
                    hidden_delta = hidden_node_deltas[j]
                    new_weight = hidden_node.weights[i]
                    delta += new_weight * hidden_delta

            prior_output = input_node.prior_output
            delta *= prior_output * (1 - prior_output)
            input_node.update(delta)

