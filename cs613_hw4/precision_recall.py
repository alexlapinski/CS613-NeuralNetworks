import binary_ann
import numpy as np


def execute(dataframe):
    # Now let's vary that threshold t (which was previously 0.5) from 0.0 to 1.0 in increments of 0.1,
    # each time computing the precision and recall by labeling observations as Positive if their likelihood
    # is greater that the current t and Negative otherwise.
    threshold_values = np.arange(0.0, 1.0, 0.1)

    num_inputs = len(dataframe.columns[:-1])

    for threshold in threshold_values:
        ann = binary_ann.BinaryANN(num_inputs, threshold=threshold)
        ann.execute(dataframe)

