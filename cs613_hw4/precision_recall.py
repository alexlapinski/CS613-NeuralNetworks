import binary_ann
import numpy as np
import pandas as pd


def execute(dataframe):
    # Now let's vary that threshold t (which was previously 0.5) from 0.0 to 1.0 in increments of 0.1,
    # each time computing the precision and recall by labeling observations as Positive if their likelihood
    # is greater that the current t and Negative otherwise.
    threshold_values = np.arange(0.0, 1.0, 0.1)

    num_inputs = len(dataframe.columns[:-1])

    metrics = pd.DataFrame(index=[i for i in xrange(len(threshold_values))],
                           columns=('threshold', 'precision', 'recall'))
    for i in xrange(len(threshold_values)):
        threshold = threshold_values[i]
        print "Executing BinaryANN using Threshold = {0}".format(threshold)
        ann = binary_ann.BinaryANN(num_inputs, threshold=threshold)
        ann.execute(dataframe)
        metrics.loc[i] = {'threshold': threshold,
                          'precision': ann.metrics.calculate_precision(),
                          'recall': ann.metrics.calculate_recall()}

    return metrics
