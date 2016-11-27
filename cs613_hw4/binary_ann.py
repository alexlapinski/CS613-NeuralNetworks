import util
import ann
import metrics
import numpy as np


class BinaryANN(object):

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
                                num_output_nodes=1,
                                learning_rate=0.5)
        self._metrics = metrics.BinaryClassificationMetrics()

    @staticmethod
    def __select_features(dataframe):
        return dataframe[dataframe.columns[:-1]]

    @staticmethod
    def __select_target_labels(dataframe):
        return dataframe[dataframe.columns[-1]]

    @property
    def metrics(self):
        """
        The metrics of how this problem performed during the call to 'execute'
        :return: BinaryClassificationMetrics
        """
        return self._metrics

    def __update_metrics(self, expected_output, actual_output):
        # We can't just apply a normal python function to a numpy array
        # This will create a ufunc to apply the given lambda to each value,
        # and map anything greater than the threshold to '1' Corresponding to the 'Is Spam' class
        apply_threshold = np.frompyfunc(lambda x: 1 if x > self._threshold else 0, 1, 1)
        formatted_actual_output = apply_threshold(actual_output)

        formatted_expected_output = expected_output.values.reshape(-1, 1)

        expected_positive_indices, _ = np.where(formatted_expected_output == 1)
        expected_negative_indices, _ = np.where(formatted_expected_output == 0)

        # True Positives are where the expected was '1' and the actual was '1'
        self._metrics.num_true_positives = np.count_nonzero(formatted_actual_output[expected_positive_indices])

        # False Negatives are where the expected was '1' and the actual was '0'
        # We can find this by subtracting the count of where actual was 1 from
        # total number where actual should be 1 (len(expected_positive_indices))
        self._metrics.num_false_negatives = len(expected_positive_indices) - self._metrics.num_true_positives

        # False negatives are where the expected was '0' and the actual was '1'
        self._metrics.num_false_positives = np.count_nonzero(formatted_actual_output[expected_negative_indices])

        # True Negatives are where the expected was '0' and the actual was '0'
        # We can compute this by subtracting the count of false positives
        # from the total number of items that should be '0'
        self._metrics.num_true_negatives = len(expected_negative_indices) - self._metrics.num_false_positives

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
        expected_training_outputs = self.__select_target_labels(training_data).values.reshape(-1, 1)
        print "Training Neural Network"
        training_errors = self._network.train(standardized_training_data, expected_training_outputs, self._iterations)

        # 7. Classifies the testing data using the trained neural network.
        print "Classifying Testing Data"
        expected_test_output = self.__select_target_labels(test_data)
        std_test_data, _, _ = util.standardize_data(self.__select_features(test_data), mean, std)

        actual_test_output = self._network.evaluate(std_test_data.values)

        # 8. Compute the testing error.
        print "Computing Metrics"
        self.__update_metrics(expected_test_output, actual_test_output)
        test_error = self._metrics.calculate_error()
        print "Test Error: ", test_error

        return test_error, training_errors
