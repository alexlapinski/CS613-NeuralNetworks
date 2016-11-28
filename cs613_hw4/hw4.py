import argparse
import matplotlib.pyplot as plt
import data
import binary_ann
import multi_ann
import precision_recall
import graphing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS 613 - HW 4 Assignment")
    parser.add_argument("--binary-ann", action="store_true", dest="do_binary_ann",
                        help="Execute the 'Binary Artificial Neural Network' problem")
    parser.add_argument("--precision-recall", action="store_true", dest="do_precision_recall",
                        help="Execute the 'Precision-Recall' problem")
    parser.add_argument("--multi-ann", action="store_true", dest="do_multi_ann",
                        help="Execute the 'Multi-Class Artificial Neural Network' problem")

    parser.add_argument("--style", action="store", dest="style", default="ggplot",
                        help="Set the matplotlib render style (default: ggplot)")
    parser.add_argument("--data", action="store", dest="data_filepath", type=str,
                        help="Set the filepath of the data csv file.")

    args = parser.parse_args()

    if not args.do_binary_ann and not args.do_precision_recall and not args.do_multi_ann:
        parser.print_help()
        exit(-1)

    if (args.do_binary_ann or args.do_precision_recall) and not args.data_filepath:
        args.data_filepath = "./data/spambase.data"

    if args.do_multi_ann and not args.data_filepath:
        args.data_filepath = "./data/CTG.csv"

    plt.style.use(args.style)

    print "Reading Data from '{0}'".format(args.data_filepath)

    graphing_helper = graphing.GraphingHelper()

    if args.do_binary_ann:
        raw_data = data.read_spambase_dataset(args.data_filepath)
        print "Executing Binary Artificial Neural Network"
        num_inputs = len(raw_data.columns[:-1])
        test_error, training_errors = binary_ann.BinaryANN(num_inputs).execute(raw_data)
        print "Plotting Graph of Training Errors"
        graph_filepath = graphing_helper.plot_binary_ann_errors(training_errors)
        print "Saved Graph to '{0}'".format(graph_filepath)
        print ""

    if args.do_precision_recall:
        raw_data = data.read_spambase_dataset(args.data_filepath)
        print "Executing Precision-Recall Problem"
        metrics = precision_recall.execute(raw_data)
        print "Plotting Precision-Recall Graph"
        graph_filepath = graphing_helper.plot_precision_recall(metrics)
        print "Saved Graph to '{0}'".format(graph_filepath)
        print ""

    if args.do_multi_ann:
        raw_data = data.read_cardiotocography_dataset(args.data_filepath)
        print "Executing Multi-Class Artificial Neural Network"
        num_inputs = len(raw_data.columns[:-1])
        test_error, training_errors = multi_ann.MultiANN(num_inputs).execute(raw_data)
        print "Plotting Graph of Training Errors"
        graph_filepath = graphing_helper.plot_multi_ann_errors(training_errors)
        print "Saved Graph to '{0}'".format(graph_filepath)
        print ""
