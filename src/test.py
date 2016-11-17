from ann import ArtificialNeuralNetwork
import numpy as np


def print_node_weights(label, nodes):
    for i in xrange(len(nodes)):
        node = nodes[i]
        print "{0}_{1} Weights: {2}".format(label, i, node.weights)

if __name__ == "__main__":
    num_inputs = 4
    num_input_nodes = 4
    num_hidden_nodes = 2
    num_output_nodes = 1
    a1 = ArtificialNeuralNetwork(num_inputs, num_input_nodes,
                                 num_hidden_nodes, num_output_nodes)

    inputs = np.array([1, 2, 2, 1])
    expected_output = 0.6
    output = 0.0

    iterations = 0
    while abs(output - expected_output) > 0.001 and iterations < 1000:
        output = a1.evaluate(inputs)[0]
        #print "Inputs: {0}".format(inputs)
        #print "Actual Output: {0}, Expected Output: {1}".format(output, expected_output)

        #print_node_weights("Input", a1.input_nodes)
        #print_node_weights("Hidden", a1.input_nodes)
        #print_node_weights("Output", a1.output_nodes)

        a1.update([expected_output])
        iterations += 1

    print "Final Output ", output
    print "Final Weights"
    print_node_weights("Input", a1.input_nodes)
    print_node_weights("Hidden", a1.hidden_nodes)
    print_node_weights("Output", a1.output_nodes)
    print "Iterations ",iterations