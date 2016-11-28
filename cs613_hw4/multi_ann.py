import util
import ann
import metrics
import numpy as np


class MultiANN(object):

    def __init__(self, num_inputs, threshold=0.5, num_hidden_nodes=20, training_data_ratio=2.0/3.0, iterations=1000):
        """
        Create an instance of th BinaryANN class used to execute the Binary ANN Problem
        :param num_hidden_nodes: Number of hidden nodes to use in ANN
        :param threshold: The minimum threshold value to consider output as 'class 1' (i.e. spam).
        :param training_data_ratio: ratio of the input data to use as training data
        :param iterations: The total number of iterations to train the ANN
        """
        self._training_data_ratio = training_data_ratio
        self._iterations = iterations
        self._threshold = threshold
        self._network = ann.ANN(num_inputs=num_inputs,
                                num_hidden_nodes=num_hidden_nodes,
                                num_output_nodes=3,
                                learning_rate=0.5)

    @staticmethod
    def __select_features(dataframe):
        return dataframe[dataframe.columns[:-1]]

    @staticmethod
    def __select_target_labels(dataframe):
        return dataframe[dataframe.columns[-1]]

    @staticmethod
    def __compute_error(expected, actual):
        # TODO: Pick the max value from each output node to assign a 'class'
        return 0

    @staticmethod
    def __compute_expected_outputs(class_labels):
        """
        Take the outputs (represented as class labels) and produce individual probabilities for each output node
        We have 3 output nodes, each represents the probability of a given class.
        For example, class 1 => [1, 0, 0], Class 2 => [0, 1, 0]
        :param class_labels: A single column-vector of class labels
        :return: a 3 x N matrix, one column per class label probability
        """

        # TODO

    def execute(self, dataframe):
        """
        Execute the Binary-Artificial Neural Network problem
        :param dataframe: Input raw data
        :return: (final test error, list of training errors for each training iteration)
        """

        # 2. Randomizes the data.
        print "Randomizing Data"
        random_data = util.randomize_data(dataframe)

        # 3. Selects the first 2/3 (round up) of the data for training and the remaining for testing
        print "Splitting Test and Training Data"
        training_data, test_data = util.split_data(random_data, self._training_data_ratio)

        # 4. Standardizes the data (except for the last column of course as well as the bias feature)
        #    using the training data
        print "Standardizing Training Data"
        standardized_training_data, mean, std = util.standardize_data(self.__select_features(training_data))

        # 5. Trains an artificial neural network using the training data
        #    Our last column is the label column
        # 6. During the training process, compute the training error after each iteration.
        #    You will use this to plot the training error vs. iteration number.
        expected_training_output_labels = self.__select_target_labels(training_data).values.reshape(-1, 1)
        expected_training_outputs = self.__compute_expected_outputs(expected_training_output_labels)
        print "Training Neural Network"
        training_errors = self._network.train(standardized_training_data, expected_training_outputs, self._iterations)

        # 7. Classifies the testing data using the trained neural network.
        print "Classifying Testing Data"
        expected_test_output = self.__select_target_labels(test_data)
        std_test_data, _, _ = util.standardize_data(self.__select_features(test_data), mean, std)

        actual_test_output = self._network.evaluate(std_test_data.values)

        # 8. Compute the testing error.
        print "Computing Metrics"
        test_error = self.__calculate_error(expected_test_output, actual_test_output)
        print "Test Error: ", test_error

        return test_error, training_errors
