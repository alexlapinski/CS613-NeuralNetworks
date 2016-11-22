import math


def split_features_target(dataset):
    """
    Return the target (last column) and features (every column but the last column) as separate dataframes
    :param dataset: Dataset where the last column is the class
    :return: Tuple of features and target, first element is the dataframe of features,
             second is the dataframe of targets
    """
    return dataset[dataset.columns[:-1]], dataset[dataset.columns[-1]]


def standardize_data(dataframe, mean = None, std = None):
    """
    Standardize the given dataframe (subtract mean and divide by standard deviation).
    No columns or rows will be excluded except for the index and headers.
    The mean and standard deviation used in standardizing the data will be returned as well.
    :param dataframe: The dataframe containing data to standardize
    :param mean: A pre-calculated mean to use
    :param std: A pre-calculated standard deviation to use
    :return: Standardized dataframe, mean used and standard deviation
    """

    if mean is None:
        mean = dataframe.mean()

    if std is None:
        std = dataframe.std()

    standardized_dataframe = (dataframe - mean) / std

    return standardized_dataframe, mean, std


def randomize_data(dataframe):
    """
    Randomize the given dataframe.
    :param dataframe:
    :return: randomized dataframe
    """
    random_seed = 0
    return dataframe.sample(n=len(dataframe), random_state=random_seed)


def split_data(dataframe, training_data_ratio):
    # type: (DataFrame, float) -> (Dataframe, DataFrame)
    """
    Split the given dataframe into 2 sets, one for training, and one for testing.
    :rtype: (DataFrame, DataFrame)
    :param dataframe: The source data to segment into training and test data
    :param training_data_ratio: The ration (e.g. 2/3) of the overall data to use for the training set, remaining
                                is used for test set.
    :return: TrainingData, TestData
    """
    if training_data_ratio > 1 or training_data_ratio < 0:
        raise ValueError("training_data_ratio must be between 0.0 and 1.0")

    max_training_index = int(math.floor(len(dataframe) * training_data_ratio))
    print "Using {0} percent of the input data for training.".format(training_data_ratio)

    training_data = dataframe.iloc[:max_training_index]
    test_data = dataframe.iloc[max_training_index:]

    print "Size of Training Data: {0}, Size of Test Data: {1}".format(len(training_data), len(test_data))

    return training_data, test_data
