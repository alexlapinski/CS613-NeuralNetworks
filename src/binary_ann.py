import util


def execute(dataframe, threshold=0.5, training_data_ratio=2.0/3):
    """
    Execute the Binary-Artificial Neural Network problem
    :param dataframe: Input raw data
    :param threshold: The threshold value to consider output as 'class 1' (i.e. spam).
    :param training_data_ratio: ratio of the input data to use as training data
    :return: BinaryMetrics
    """
    learning_parameter = 0.5
    training_iterations = 1000

    # 2. Randomizes the data.
    random_data = util.randomize_data(dataframe)

    # 3. Selects the first 2/3 (round up) of the data for training and the remaining for testing
    training_data, test_data = util.split_data(random_data, training_data_ratio)

    # 4. Standardizes the data (except for the last column of course as well as the bias feature)
    #    using the training data

    # 5. Trains an artificial neural network using the training data

    # 6. During the training process, compute the training error after each iteration.
    #    You will use this
    #    to plot the training error vs. iteration number.

    # 7. Classifies the testing data using the trained neural network.

    # 8. Computes the testing error.

    # Implementation Details
    # 2. Make sure to add a bias input node.
    # 3. Set the learning parameter  = 0.5.
    # 4. There should only be a single hidden layer.
    # 5. The hidden layer size should be 20, although this should be a variable parameter.
    # 6. Do batch gradient descent.
    # 7. Initialize all weights to random values in the range [-1; 1].
    # 8. Do 1000 training iterations.
    # 9. Since this is binary classification, you should only have one output node.
    # 10. Consider a sample to be positive (Spam) if the output node has a value > 0:50
    #     and negative (Not Spam) otherwise.
    # 11. Compute the testing error as
    #     TestError = 1 - (# test samples correct/ total # of test samples)
    pass
