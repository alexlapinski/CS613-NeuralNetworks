from .context import cs613_hw4 as cs
import numpy as np
import pytest


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

    is_class_1 = network.evaluate(inputs)
    print "Test Evaluation, is_class_1:", is_class_1


def test_train_multiple_outputs():
    inputs = np.array([1, 2, 3])
    expected_outputs = np.array([1, 0])
    num_inputs = 3
    num_hidden_nodes = 3
    num_output_nodes = 2
    network = cs.ann.SingleANN(num_inputs, num_hidden_nodes, num_output_nodes)
    network.train(inputs, expected_outputs, verbose=True)

    print "Final Hidden Weights:"
    print network.hidden_weights
    print ""
    print "Final Output Weights:"
    print network.output_weights
    print ""

    is_class_1, is_class_2 = network.evaluate(inputs)
    print "Test Evaluation, is_class_1:{0}, is_class_2:{1}".format(is_class_1, is_class_2)


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

    is_class_1 = network.evaluate(np.array([1, 2, 3]))
    print "Test Evaluation, input: {0}, is_class_1: {1}".format(np.array([1, 2, 3]), is_class_1)
