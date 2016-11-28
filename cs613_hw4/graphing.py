import matplotlib.pyplot as plt
import os


class GraphingHelper(object):

    @staticmethod
    def __ensure_dir_exists(directory):
        if not os.path.exists(directory):
            os.mkdirs(directory)

    def __init__(self, output_dir='graphs'):
        """
        Helper for creating graphs.
        :param output_dir: Path to a directory where the graph will be stored, will be created if it does not exist
        """

        self._output_dir = output_dir
        self.__ensure_dir_exists(self._output_dir)

    def plot_training_ann_errors(self, training_errors, title=None, filename='ann_training_error.png'):
        """
        Plot the training errors for each iteration.
        :param training_errors: list of errors for each training iteration
        :param filename: Graph filename
        :return: full path to the created image
        """

        plt.plot(training_errors, color='r')
        plt.ylabel("Error")
        plt.xlabel("Iteration")

        if title is not None:
            plt.title(title)

        graph_filepath = os.path.join(self._output_dir, filename)
        plt.savefig(graph_filepath)

        return graph_filepath

    def plot_precision_recall(self, metrics, filename='precision_recall.png'):
        """
        Plot the precision-recall graph produced by varying the threshold to the BinaryANN
        :param metrics: dataframe of precision-recall metrics
        :param filename: Graph filename
        :return: full path to the created image
        """

        plt.plot(metrics['precision'], metrics['recall'], color='b')
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        plt.title('SPAM Detection')

        graph_filepath = os.path.join(self._output_dir, filename)
        plt.savefig(graph_filepath)

        return graph_filepath
