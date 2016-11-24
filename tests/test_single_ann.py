from .context import cs613_hw4 as cs
import numpy as np
import pytest


@pytest.mark.skip(reason="Not Ready Yet")
def test_forward_propagation():
    inputs = np.array([1, 2, 3])
    expected_outputs = np.array([1])
    num_inputs = 3
    num_hidden_nodes = 2
    num_output_nodes = 1
    network = cs.ann.SingleANN(num_inputs, num_hidden_nodes, num_output_nodes)
    actual_output = network.evaluate(inputs, True)

    print "Expected Outputs:", expected_outputs
    print "Actual Outputs:", actual_output


def test_train():
    inputs = np.array([1, 2, 3])
    expected_outputs = np.array([1])
    num_inputs = 3
    num_hidden_nodes = 3
    num_output_nodes = 1
    network = cs.ann.SingleANN(num_inputs, num_hidden_nodes, num_output_nodes)
    network.train(inputs, expected_outputs, verbose=True)

    print "Final Hidden Weights:"
    print network.hidden_weights
    print ""
    print "Final Output Weights:"
    print network.output_weights
    print ""


def test_train_multiple_samples():
    num_inputs = 3
    num_hidden_nodes = 3
    num_output_nodes = 1
    network = cs.ann.SingleANN(num_inputs, num_hidden_nodes, num_output_nodes)

    inputs = np.array([1, 2, 3])
    expected_outputs = np.array([1])
    network.train(inputs, expected_outputs, verbose=True)

    inputs = np.array([0.1, 3, 2])
    expected_outputs = np.array([0])
    network.train(inputs, expected_outputs, verbose=True)

    inputs = np.array([1, 2, 3])
    expected_outputs = np.array([1])
    network.train(inputs, expected_outputs, verbose=True)

    inputs = np.array([1, 0, 1])
    expected_outputs = np.array([1])
    network.train(inputs, expected_outputs, verbose=True)

    print "Final Hidden Weights:"
    print network.hidden_weights
    print ""
    print "Final Output Weights:"
    print network.output_weights
    print ""
