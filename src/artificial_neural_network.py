from perceptron import Perceptron
import numpy as np


class ArtificialNeuralNetwork:

    @staticmethod
    def _create_nodes(num_inputs, num_nodes):
        nodes = []
        for i in xrange(num_nodes):
            nodes.append(Perceptron(num_inputs))

        return nodes

    def __init__(self, num_inputs, num_input_nodes, num_hidden_nodes, num_output_nodes):
        self._input_nodes = self._create_nodes(num_inputs, num_input_nodes)

        if num_hidden_nodes == 0:
            self._hidden_nodes = None
            self._output_nodes = self._create_nodes(num_input_nodes, num_output_nodes)
        else:
            self._hidden_nodes = self._create_nodes(num_input_nodes, num_hidden_nodes)
            self._output_nodes = self._create_nodes(num_hidden_nodes, num_output_nodes)



    def evaluate(self, inputs):

        hidden_inputs = []
        for input_node in self._input_nodes:
            hidden_inputs.append(input_node.evaluate(inputs))

        print hidden_inputs

        if self._hidden_nodes is None:
            output_inputs = hidden_inputs
        else:
            output_inputs = []
            for hidden_node in self._hidden_nodes:
                output_inputs.append(hidden_node.evaluate(np.array(hidden_inputs)))

        print output_inputs

        final_outputs = []
        for output_node in self._output_nodes:
            final_outputs.append(output_node.evaluate(np.array(output_inputs)))

        return final_outputs