"""
tests for churn_library

author: Erick Alvarez
date: July 19 2024
"""
import os
import logging
import pytest
import pandas as pd
import joblib
import churn_library as cls
import constants


@pytest.fixture
def test_df():
    """
        fixture for the dataset for testing
    """
    return pd.DataFrame(constants.TEST_DATA, columns=constants.ALL_COL_NAMES)


@pytest.fixture
def feature_engineered_set(test_df):
    """
        fixture for the sets after performing feature engineering
    """
    X_train_set, X_test_set, y_train_set, y_test_set = cls.perform_feature_engineering(
        test_df)
    return X_train_set, X_test_set, y_train_set, y_test_set


def test_import():
    """
        test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = cls.import_data(constants.BANK_DATA_PTH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(test_df):
    """
        test perform eda function
        """
    try:
        cls.perform_eda(test_df, constants.EDA_IMGS_TEST_PTH)
        logging.info("Testing perform_eda: SUCCESS running the method")
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_eda: The directory %s wasn't found",
            constants.EDA_IMGS_TEST_PTH)
        raise err

    # verify that all distribution graphs exist
    for col_name in constants.ALL_COL_NAMES:
        try:
            graph_path = os.path.join(
                constants.EDA_IMGS_TEST_PTH,
                col_name + " distribution.png")
            assert os.path.exists(graph_path)
        except AssertionError as err:
            logging.error(
                "Testing perform_eda: The distribution graph file %s wasn't found",
                graph_path)
            raise err
    logging.info("Testing perform_eda: SUCCESS all distribution graphs exist")

    # verify that the correlation matrix graph exists
    try:
        graph_path = os.path.join(
            constants.EDA_IMGS_TEST_PTH,
            "correlation matrix.png")
        assert os.path.exists(graph_path)
        logging.info(
            "Testing perform_eda: SUCCESS correlation matrix graph exists")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The correlation matrix graph file %s wasn't found",
            graph_path)
        raise err


def test_encoder_helper(test_df):
    """
        test encoder helper
        """

    expected_one_hot_col_names = set()
    for col_name, val_list in constants.CATEGORY_VALS_DICT.items():
        for val in val_list:
            expected_one_hot_col_names.add(f"{col_name}_{val}")

    one_hot_test_df, one_hot_col_names = cls.encoder_helper(
        test_df, constants.CAT_COLUMNS)

    # Test that the column names are correct
    try:
        assert expected_one_hot_col_names == set(one_hot_col_names)
        logging.info(
            "Testing encoder_helper: SUCCESS the names of the categorical columns are correct")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: the names of the categorical columns are not correct")
        raise err

    # Test that the one hot encoding is correct

    # Iterate over all categorical columns
    for cat_col in constants.CAT_COLUMNS:
        # iterate over all entries in the respective column
        for el_index, el_cat_val in test_df[cat_col].iteritems():
            # check all possible values for this category. If the value matches
            # with entry, then expect a 1; otherwise a 0
            for val in constants.CATEGORY_VALS_DICT[cat_col]:
                one_hot_cat_col = f"{cat_col}_{val}"
                actual_encoding = one_hot_test_df.loc[el_index,
                                                      one_hot_cat_col]
                if val == el_cat_val:
                    expected_encoding = 1
                else:
                    expected_encoding = 0
                try:
                    assert expected_encoding == actual_encoding
                except AssertionError as err:
                    logging.error(
                        "Testing encoder_helper: the encoding of %s is wrong. \
                        The encoding should be %s but %s was found",
                        one_hot_cat_col,
                        expected_encoding,
                        actual_encoding)
                    raise err

    logging.info(
        "Testing encoder_helper: SUCCESS the one hot encoding is correct")


def test_perform_feature_engineering(feature_engineered_set):
    """
        test perform_feature_engineering
        """
    X_train, X_test, y_train, y_test = feature_engineered_set

    # verify splits are not empty
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info(
            "Testing perform_feature_engineering: SUCCESS all dataframes contain data")

    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: a dataframe is empty")
        raise err

    # verify that sizes of dataframes match
    try:
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        logging.info(
            "Testing perform_feature_engineering: SUCCESS the sizes of the training \
            dataframes and test dataframes are correct")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The size of either the training \
            sets ot the testing sets do not match")
        raise err


def test_train_models(feature_engineered_set):
    """
        test train_models
        """
    # just to increase size of the data
    X_train_set, X_test_set, y_train_set, y_test_set = feature_engineered_set
    cls.train_models(
        X_train_set,
        X_test_set,
        y_train_set,
        y_test_set,
        constants.MODELS_TEST_PTH,
        constants.RES_IMGS_TEST_PTH)

    # Verify that models were saved
    models_pths = [
        f'{constants.MODELS_TEST_PTH}rfc_model.pkl',
        f'{constants.MODELS_TEST_PTH}logistic_model.pkl']
    for model_pth in models_pths:
        try:
            joblib.load(model_pth)
        except FileNotFoundError as err:
            logging.error(
                "Testing train_models: The model file %s wasn't found",
                model_pth)
            raise err
    logging.info("Testing train_models: SUCCESS all models saved successfully")

    # Verify that all result plots where created
    plot_pths = [
        f'{constants.RES_IMGS_PTH}{plot_name}' for plot_name in constants.RESULTS_PLOTS_NAMES]
    for plot_pth in plot_pths:
        try:
            assert os.path.exists(plot_pth)
        except AssertionError as err:
            logging.error(
                "Testing train_models: The plot %s wasn't found",
                plot_pth)
            raise err
    logging.info("Testing train_models: SUCCESS all result plots found")
