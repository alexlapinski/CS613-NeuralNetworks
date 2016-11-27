class BinaryClassificationMetrics(object):
    def __init__(self):
        self._num_true_positives = 0
        self._num_true_negatives = 0
        self._num_false_positives = 0
        self._num_false_negatives = 0

    @property
    def num_true_positives(self):
        return self._num_true_positives

    @num_true_positives.setter
    def num_true_positives(self, value):
        self._num_true_positives = value

    @property
    def num_true_negatives(self):
        return self._num_true_negatives

    @num_true_negatives.setter
    def num_true_negatives(self, value):
        self._num_true_negatives = value

    @property
    def num_false_positives(self):
        return self._num_false_positives

    @num_false_positives.setter
    def num_false_positives(self, value):
        self._num_false_positives = value

    @property
    def num_false_negatives(self):
        return self._num_false_positives

    @num_false_negatives.setter
    def num_false_negatives(self, value):
        self._num_false_negatives = value

    def calculate_precision(self):
        """
        Compute the Precision (percentage of things that were classified as positive and actually were positive)
        :return: precision, value between 0.0 and 1.0
        """
        return self._num_true_positives / (self._num_true_positives + self._num_false_positives)

    def calculate_recall(self):
        """
        Compute the Recall (True Positive Rate),
        the percentage of true positives (sensitivity) which were correctly identified.
        :return: recall, value between 0.0 and 1.0
        """
        return self._num_true_positives / (self._num_true_positives + self._num_false_negatives)

    def calculate_error(self):
        """
        Compute the Error for Binary Classification
        :return: float value between 0.0 and 1.0
        """

        num_correct = self._num_true_positives + self._num_true_negatives
        num_incorrect = self._num_false_positives + self._num_false_negatives
        num_samples = num_correct + num_incorrect

        return 1.0 - (num_correct / num_samples)
