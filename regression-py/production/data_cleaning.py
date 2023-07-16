"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning
)
from scripts import binned_selling_price


@register_processor("data-cleaning", "Housing")
def clean_Housing_table(context, params):
    """Clean the ``Housing`` data table.
    """

    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"

    # load dataset
    house_df = load_dataset(context, input_dataset)

    house_df_clean = (
    house_df
    # while iterating on testing, it's good to copy the dataset(or a subset)
    # as the following steps will mutate the input dataframe. The copy should be
    # removed in the production code to avoid introducing perf. bottlenecks.
    .copy()
    # set dtypes : nothing to do here
    .passthrough()
    .transform_columns(['ocean_proximity'], string_cleaning, elementwise=False)
    .replace({'': np.NaN})
    # clean column names (comment out this line while cleaning data above)
    .clean_names(case_type='snake'))

    # save the dataset
    save_dataset(context, house_df_clean, output_dataset)

    return house_df_clean

@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``HOUSING`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/housing"
    # load dataset
    sales_df_processed = load_dataset(context, input_dataset)

    target_col = "median_house_value"

    train_X, train_y = (
        house_df_train
        
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )
    save_dataset(context, train_X, 'train/housing/features')
    save_dataset(context, train_y, 'train/housing/target')


    test_X, test_y = (
        house_df_test
        
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )
    save_dataset(context, test_X, 'test/housing/features')
    save_dataset(context, test_y, 'test/housing/target')

