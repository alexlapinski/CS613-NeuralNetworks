import util
import ann
import numpy as np


def __select_features(dataframe):
    return dataframe[dataframe.columns[:-1]]


def __select_target_labels(dataframe):
    return dataframe[dataframe.columns[-1]]


def execute(dataframe, threshold=0.5, num_hidden_nodes=20, training_data_ratio=2.0/3, iterations=1000):
    """
    Execute the Binary-Artificial Neural Network problem
    :param num_hidden_nodes: Number of hidden nodes to use in ANN
    :param dataframe: Input raw data
    :param threshold: The minimum threshold value to consider output as 'class 1' (i.e. spam).
    :param training_data_ratio: ratio of the input data to use as training data
    :param iterations: The total number of iterations to train the ANN
    :return: (final test error, list of training errors for each training iteration)
    """
    learning_parameter = 0.5

    # 2. Randomizes the data.
    print "Randomizing Data"
    random_data = util.randomize_data(dataframe)

    # 3. Selects the first 2/3 (round up) of the data for training and the remaining for testing
    print "Splitting Test and Training Data"
    training_data, test_data = util.split_data(random_data, training_data_ratio)

    # 4. Standardizes the data (except for the last column of course as well as the bias feature)
    #    using the training data
    print "Standardizing Training Data"
    standardized_training_data, mean, std = util.standardize_data(__select_features(training_data))

    # 5. Trains an artificial neural network using the training data
    #    Our last column is the label column
    num_inputs = len(dataframe.columns[:-1])

    # Network will add the bias internally
    network = ann.ANN(num_inputs, num_hidden_nodes, num_output_nodes=1, learning_rate=learning_parameter)

    # 6. During the training process, compute the training error after each iteration.
    #    You will use this to plot the training error vs. iteration number.
    expected_training_outputs = __select_target_labels(training_data).values.reshape(-1, 1)
    print "Training Neural Network"
    training_errors = network.train(standardized_training_data, expected_training_outputs, iterations, threshold=threshold)

    # 7. Classifies the testing data using the trained neural network.
    print "Classifying Testing Data"
    expected_test_output = __select_target_labels(test_data)
    std_test_data, _, _ = util.standardize_data(__select_features(test_data), mean, std)

    raw_actual_test_output = network.evaluate(std_test_data.values)

    # We can't just apply a normal python function to a numpy array
    # This will create a ufunc to apply the given lambda to each value, and map anything greater than the threshold to
    # '1' Corresponding to the 'Is Spam' class
    apply_threshold = np.frompyfunc(lambda x: 1 if x > threshold else 0, 1, 1)
    actual_test_output = apply_threshold(raw_actual_test_output)

    # 8. Compute the testing error.
    print "Computing test error"
    # Count all of the true values (where expected == actual)
    num_correct = np.count_nonzero(expected_test_output.values.reshape(-1, 1) == actual_test_output)
    num_samples = len(expected_test_output)
    test_error = 1.0 - (float(num_correct) / float(num_samples))

    print "Test Error: ", test_error

    return test_error, training_errors