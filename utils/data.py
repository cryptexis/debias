import tensorflow as tf
from tensorflow import feature_column


def convert_categorical(col_name, cat_maps):
    """
    :param col_name:
    :param cat_maps:
    :return: one_hot_encoded column
    """
    feature_col = feature_column.categorical_column_with_vocabulary_list(col_name, cat_maps[col_name])
    col_one_hot = feature_column.indicator_column(feature_col)

    return col_one_hot


def categorical_dicts(df, list_columns):
    """
    Creates dictionary key=column_name, value=unique_values in the column
    :param df: Pandas DataFrame
    :param list_columns: list of column names
    :return: dictionary
    """

    col_maps = {}
    for col in list_columns:
        col_maps[col] = list(df[col].unique())
    return col_maps


def make_input_fn(dataframe, labels, shuffle=True, batch_size=32):
    """
    Data Input pipeline
    :param dataframe: Pandas dataframe
    :param labels: Labels column
    :param shuffle: flag whether to shuffle the data
    :param batch_size: number of datapoints in batch
    :return:
    """
    def df_to_dataset():

        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))

        ds = ds.batch(batch_size)
        return ds

    return df_to_dataset


def breakdown_df_shapes(global_tp, global_tn, global_fp, global_fn,  col, val):
    """
    Counts tp, fp, fn, tn on a subset
    :param global_tp: Dataframe with all true_positives
    :param global_tn: Dataframe with all true_negatives
    :param global_fp: Dataframe with all false_positives
    :param global_fn: Dataframe with all false_negatives
    :param col: Candidate column
    :param val: Value of the column
    :return: tuple of of (tp, fp, fn, tn) computed on the subset
    """
    return (global_tp[global_tp[col] == val].shape[0],
            global_tn[global_tn[col] == val].shape[0],
            global_fp[global_fp[col] == val].shape[0],
            global_fn[global_fn[col] == val].shape[0])


def breakdown_metrics(global_tp, global_tn, global_fp, global_fn,  col, val):
    """
    Calculates FPR annd FNR on a subset
    :param global_tp: Dataframe with all true_positives
    :param global_tn: Dataframe with all true_negatives
    :param global_fp: Dataframe with all false_positives
    :param global_fn: Dataframe with all false_negatives
    :param col: Candidate column
    :param val: Value of the column
    :return:
    """
    tp_local, tn_local, fp_local, fn_local = breakdown_df_shapes(global_tp, global_tn, global_fp, global_fn, col, val)

    fpr = float(fp_local)/(fp_local + tn_local)

    fnr = float(fn_local)/(fn_local + tp_local)

    print(f"For value {val}: FPR: {fpr}, FNR: {fnr}")

    return fpr, fnr

