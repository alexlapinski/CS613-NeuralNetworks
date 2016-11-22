import pandas as pd


def read_spambase_dataset(filepath):
    """
    Read the spambase data file at the given filepath and return a dataframe
    :param filepath: filepath of the spambase data file
    :return: Pandas dataframe, where the last column is the class of the row (1 = spam, 0 = not spam)
    """

    # Columns (imported from spambase.names file)
    column_names = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
                    "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
                    "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report",
                    "word_freq_addresses", "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you",
                    "word_freq_credit", "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
                    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab",
                    "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415",
                    "word_freq_85", "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm",
                    "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
                    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;",
                    "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_  #",
                    "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total", "Is Spam"]

    # The id column is always the last in the list of columns
    return pd.read_csv(filepath, names=column_names)


def read_cardiotocography_dataset(filepath):
    """
    Read the Cardiotocography CSV.
    The first row is the header, and the second row is interpreted as a data row, but dropped from the dataset.
    The 2nd to last column in the file is dropped from the returned dataset.
    The last column of the returned dataset represents the class of the row.
    :param filepath: filepath of the Cardiotocography database
    :return: Pandas dataframe, where the last column is the class of the row
    """

    # The Cardiotocography dataset is in csv form
    # The first row is the header row
    # Per requirements, we should only be using the last column as the class label
    #   We also need to discard the 2nd to last column and not use it

    raw_data = pd.read_csv(filepath)

    # Remove the rows from the dataframe which are "NaN", we know the 2nd row in the file is blank
    raw_data.dropna(inplace=True)

    # Remove the 2nd to last column in the source dataset
    columns = raw_data.columns.delete(-2)
    return raw_data[columns]
