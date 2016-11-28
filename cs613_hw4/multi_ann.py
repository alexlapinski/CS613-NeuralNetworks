import util
import ann


class MultiANN(object):

    def __init__(self, num_inputs, num_hidden_nodes=20, training_data_ratio=2.0/3.0, iterations=1000, labels=(1, 2, 3)):
        """
        Create an instance of th MultiANN class used to execute the Multi-class ANN Problem
        :param num_hidden_nodes: Number of hidden nodes to use in ANN
        :param training_data_ratio: ratio of the input data to use as training data
        :param iterations: The total number of iterations to train the ANN
        :param labels: A tuple of labels, the first label in the tuple corresponds to the first output node
        """
        self._training_data_ratio = training_data_ratio
        self._iterations = iterations
        self._multiclass_helper = ann.MulticlassHelper(labels)
        self._network = ann.ANN(num_inputs=num_inputs,
                                num_hidden_nodes=num_hidden_nodes,
                                num_output_nodes=len(labels),
                                learning_rate=0.5)

    @staticmethod
    def __select_features(dataframe):
        return dataframe[dataframe.columns[:-1]]

    @staticmethod
    def __select_target_labels(dataframe):
        return dataframe[dataframe.columns[-1]]

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
        print "Training Neural Network"
        training_errors = self._network.train_multiclass(standardized_training_data, expected_training_output_labels,
                                                         self._iterations)

        # 7. Classifies the testing data using the trained neural network.
        print "Classifying Testing Data"
        expected_test_output = self.__select_target_labels(test_data).values.reshape(-1, 1)
        std_test_data, _, _ = util.standardize_data(self.__select_features(test_data), mean, std)

        actual_test_output = self._network.evaluate(std_test_data.values)

        # 8. Compute the testing error.
        print "Computing Metrics"
        labeled_output = self._multiclass_helper.assign_label(actual_test_output)
        test_error = self._multiclass_helper.calculate_error(expected_test_output, labeled_output)
        print "Test Error: ", test_error

        return test_error, training_errors
