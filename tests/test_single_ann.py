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
