from .context import cs613_hw4 as cs
import numpy as np
import pytest


run_exploratory_tests = False

exploratory = pytest.mark.skipif(
    not run_exploratory_tests,
    reason="Requires 'run_exploratory_tests' to be 'True'"
)


def test_forward_propagation():
    inputs = np.array([[0.6, 0.4]])
    expected_output = np.array([0.55621109888233611])
    expected_hidden_outputs = np.array([0.4650570548417855, 0.65475346060631923])

    network = cs.ann.ANN(num_inputs=2,
                         num_hidden_nodes=2,
                         num_output_nodes=1)

    network.hidden_weights = np.array([[0.4, 0.3], [-0.7, 0.9], [-0.1, 0.1]])
    network.output_weights = np.array([[-0.5], [0.7]])

    actual_output = network.evaluate(inputs)

    prior_hidden_outputs = network.prior_hidden_outputs

    assert np.allclose(prior_hidden_outputs, expected_hidden_outputs, atol=0.001)
    assert np.allclose(actual_output, expected_output, atol=0.001)


def test_forward_propagation_batch():
    inputs = np.array([[0.6, 0.4], [0.5, 0.3]])
    expected_output = np.array([[0.55621109888233611], [0.55051199790344874]])
    expected_hidden_outputs = np.array([[0.4650570548417855, 0.65475346060631923],
                                        [0.47252769565540637, 0.62714776631319558]])

    network = cs.ann.ANN(num_inputs=2,
                         num_hidden_nodes=2,
                         num_output_nodes=1)

    network.hidden_weights = np.array([[0.4, 0.3], [-0.7, 0.9], [-0.1, 0.1]])
    network.output_weights = np.array([[-0.5], [0.7]])

    actual_output = network.evaluate(inputs)

    prior_hidden_outputs = network.prior_hidden_outputs

    assert np.allclose(prior_hidden_outputs, expected_hidden_outputs, atol=0.001)
    assert np.allclose(actual_output, expected_output, atol=0.001)


def test_backward_propagation():
    inputs = np.array([[0.6, 0.4]])
    expected_output = np.array([[1]])
    network = cs.ann.ANN(num_inputs=2,
                         num_hidden_nodes=2,
                         num_output_nodes=1)

    network.hidden_weights = np.array([[0.4, 0.3], [-0.7, 0.9], [-0.1, 0.1]])
    network.output_weights = np.array([[-0.5], [0.7]])

    network.evaluate(inputs)
    network.update(expected_output)

    expected_new_output_weights = np.array([[-0.44905533], [0.77172496]])
    assert np.allclose(network.output_weights, expected_new_output_weights, atol=0.001)

    expected_new_hidden_weights = np.array([[0.39265727386632238, 0.31146604016883561],
                                            [-0.70489515075578502, 0.90764402677922373],
                                            [-0.11223787688946278, 0.11911006694805942]])

    assert np.allclose(network.hidden_weights, expected_new_hidden_weights, atol=0.001)


def test_backward_propagation_batch():
    inputs = np.array([[0.6, 0.4], [0.2, 0.7]])
    expected_output = np.array([[1], [0]])
    network = cs.ann.ANN(num_inputs=2,
                         num_hidden_nodes=2,
                         num_output_nodes=1)

    network.hidden_weights = np.array([[0.4, 0.3],
                                       [-0.7, 0.9],
                                       [-0.1, 0.1]])
    network.output_weights = np.array([[-0.5], [0.7]])

    network.evaluate(inputs)
    network.update(expected_output)

    expected_new_output_weights = np.array([[-0.50082647], [0.68764965]])
    assert np.allclose(network.output_weights, expected_new_output_weights, atol=0.001)

    expected_new_hidden_weights = np.array([[0.39755125, 0.30303854],
                                            [-0.69696917, 0.89616098],
                                            [-0.09859497, 0.09816457]])

    assert np.allclose(network.hidden_weights, expected_new_hidden_weights, atol=0.001)


def test_forward_propagation_multi_output():
    inputs = np.array([[0.6, 0.4]])
    expected_output = np.array([0.5562111, 0.50902464])
    expected_hidden_outputs = np.array([0.46505705, 0.65475346])

    network = cs.ann.ANN(num_inputs=2,
                         num_hidden_nodes=2,
                         num_output_nodes=2)

    network.hidden_weights = np.array([[0.4, 0.3], [-0.7, 0.9], [-0.1, 0.1]])
    network.output_weights = np.array([[-0.5, 0.5], [0.7, -0.3]])

    actual_output = network.evaluate(inputs)

    prior_hidden_outputs = network.prior_hidden_outputs

    assert np.allclose(prior_hidden_outputs, expected_hidden_outputs, atol=0.001)
    assert np.allclose(actual_output, expected_output, atol=0.001)


def test_backward_propagation_multi_output():
    inputs = np.array([[0.6, 0.4]])
    expected_output = np.array([[1, 0]])
    network = cs.ann.ANN(num_inputs=2,
                         num_hidden_nodes=2,
                         num_output_nodes=2)

    network.hidden_weights = np.array([[0.4, 0.3],
                                       [-0.7, 0.9],
                                       [-0.1, 0.1]])
    network.output_weights = np.array([[-0.5, 0.5], [0.7, -0.3]])

    network.evaluate(inputs)
    network.update(expected_output)

    expected_new_output_weights = np.array([[-0.44905534, 0.44083791],
                                            [0.77172495, -0.38329426]])
    assert np.allclose(network.output_weights, expected_new_output_weights, atol=0.001)

    expected_new_hidden_weights = np.array([[0.37800298, 0.31316328],
                                            [-0.71466468, 0.90877552],
                                            [-0.1366617, 0.12193879]])

    assert np.allclose(network.hidden_weights, expected_new_hidden_weights, atol=0.001)


@exploratory
def test_train():
    inputs = np.array([1, 2, 3])
    expected_outputs = np.array([1])
    num_inputs = 3
    num_hidden_nodes = 3
    num_output_nodes = 1
    network = cs.ann.ANN(num_inputs, num_hidden_nodes, num_output_nodes)
    network.train(inputs, expected_outputs, verbose=True)

    print "Final Hidden Weights:"
    print network.hidden_weights
    print ""
    print "Final Output Weights:"
    print network.output_weights
    print ""

    is_class_1 = network.evaluate(inputs)
    print "Test Evaluation, is_class_1:", is_class_1


@exploratory
def test_train_multiple_outputs():
    inputs = np.array([1, 2, 3])
    expected_outputs = np.array([1, 0])
    num_inputs = 3
    num_hidden_nodes = 3
    num_output_nodes = 2
    network = cs.ann.ANN(num_inputs, num_hidden_nodes, num_output_nodes)
    network.train(inputs, expected_outputs, verbose=True)

    print "Final Hidden Weights:"
    print network.hidden_weights
    print ""
    print "Final Output Weights:"
    print network.output_weights
    print ""

    is_class_1, is_class_2 = network.evaluate(inputs)
    print "Test Evaluation, is_class_1:{0}, is_class_2:{1}".format(is_class_1, is_class_2)


@exploratory
def test_train_multiple_samples():
    num_inputs = 3
    num_hidden_nodes = 3
    num_output_nodes = 1
    network = cs.ann.ANN(num_inputs, num_hidden_nodes, num_output_nodes)

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
