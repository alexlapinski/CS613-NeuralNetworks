from ann import ArtificialNeuralNetwork
import numpy as np


def print_node_weights(label, nodes):
    for i in xrange(len(nodes)):
        node = nodes[i]
        print "{0}_{1} Weights: {2}".format(label, i, node.weights)

if __name__ == "__main__":
    num_inputs = 4
    num_input_nodes = 5
    num_hidden_nodes = 4
    num_output_nodes = 1
    learning_rate = 1.0
    a1 = ArtificialNeuralNetwork(num_inputs, num_input_nodes,
                                 num_hidden_nodes, num_output_nodes,
                                 learning_rate)

    inputs = np.array([[1, 2, 2, 1], [2, 2, 2, 2], [1, 3, 3, 1], [2, 2, 2, 2]])
    expected_output = np.array([[1], [0], [1], [0]])
    output = np.zeros(expected_output)

    iterations = 0
    while iterations < 1000:
        output = a1.evaluate(inputs)
        a1.update(expected_output)
        iterations += 1

    print "Final Output ", output
    print "Final Weights"
    print_node_weights("Input", a1.input_nodes)
    print_node_weights("Hidden", a1.hidden_nodes)
    print_node_weights("Output", a1.output_nodes)
    print "Iterations ",iterations

    print a1.evaluate(np.array([[2, 2, 2, 2], [1, 2, 2, 1]]))