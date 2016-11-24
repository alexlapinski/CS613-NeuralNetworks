from .context import cs613_hw4
import numpy as np


def sigma(x):
    return 1.0 / (1.0 + np.exp(-1*x))


def forward_propagate(inputs, weights):
    return sigma(inputs.dot(weights))


def compute_delta(expected_output, actual_output):
    return (expected_output - actual_output) * actual_output * (1 - actual_output)


def backward_propagate(learning_rate, weights, prior_inputs, expected_output, actual_output):
    num_samples = prior_inputs.shape[0] # Stacked vertically
    delta = compute_delta(expected_output, actual_output)
    print delta
    return weights + (((learning_rate / float(num_samples)) * delta).T.dot(prior_inputs).T)


def execute_test():
    node_inputs = 3
    num_samples = 4
    inputs = np.random.randint(1, 10, size=node_inputs*num_samples).reshape(num_samples, node_inputs)
    expected_outputs = np.random.randint(0, 1, size=num_samples).reshape(-1, 1)

    weights = np.random.uniform(-1, 1, node_inputs).reshape(-1, 1)
    print "Original Weights"
    print weights
    print ""
    learning_rate = 1.0

    iteration = 0
    prior_inputs = inputs
    while iteration < 2:
        print "Iteration {0}".format(iteration)

        prior_inputs = inputs
        actual_outputs = forward_propagate(inputs, weights)

        print "Actual Outputs"
        print actual_outputs
        print ""
        print "Expected Outputs"
        print expected_outputs
        print ""

        new_weights = backward_propagate(learning_rate, weights, prior_inputs, expected_outputs, actual_outputs)
        print "New Weights"
        print new_weights
        print ""
        weights = new_weights

        iteration += 1



if __name__ == "__main__":
    execute_test()