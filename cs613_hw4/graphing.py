import matplotlib.pyplot as plt
import os


def __ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.mkdirs(directory)


def plot_binary_ann_errors(training_errors, output_dir='graphs', filename='binary_ann_training_error.png'):
    """
    Plot the training errors for each iteration.
    :param training_errors: list of errors for each training iteration
    :param output_dir: Path to a directory where the graph will be stored, will be created if it does not exist
    :param filename: Graph filename
    :return: full path to the created image
    """

    __ensure_dir_exists(output_dir)

    plt.plot(training_errors, color='r')
    plt.ylabel("Error")
    plt.xlabel("Iteration")
    plt.title("Error = 1 - (# correctly identified)(Total # of samples)", fontsize=10, color='gray')

    graph_filepath = os.path.join(output_dir, filename)
    plt.savefig(graph_filepath)

    return graph_filepath


def plot_precision_recall(metrics, output_dir='graphs', filename='precision_recall.png'):
    """
    Plot the precision-recall graph produced by varying the threshold to the BinaryANN
    :param metrics: dataframe of precision-recall metrics
    :param output_dir: path to a directory where the graph will be stored, will be created if it does not exist
    :param filename: Graph filename
    :return: full path to the created image
    """

    __ensure_dir_exists(output_dir)

    plt.plot(metrics['precision'], metrics['recall'], color='b')
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.title('SPAM Detection')

    graph_filepath = os.path.join(output_dir, filename)
    plt.savefig(graph_filepath)

    return graph_filepath