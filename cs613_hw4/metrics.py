class BinaryMetrics:

    def __init__(self, num_total_samples):
        self._test_errors_by_iteration = {}
        self._precision_by_threshold = {}
        self._recall_by_threshold = {}
        self._num_total_samples = num_total_samples

    def add_test_error(self, iteration, num_correct_samples):
        test_error = float(num_correct_samples) / self._num_total_samples
        self._test_errors_by_iteration[iteration] = test_error

    def test_errors(self):
        return self._test_errors_by_iteration

    def add_precision_recall(self, threshold_value, num_true_positives, num_false_positives, num_true_negatives):
        precision = float(num_true_positives) / (num_true_positives + num_false_positives)
        recall = float(num_true_positives) / (num_true_positives + num_true_negatives)

        self._precision_by_threshold[threshold_value] = precision
        self._recall_by_threshold[threshold_value] = recall

    def precision(self):
        return self._precision_by_threshold

    def recall(self):
        return self._recall_by_threshold
