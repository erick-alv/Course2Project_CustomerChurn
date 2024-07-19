"""
module to predict the customer churn

author: Erick Alvarez
date: July 19 2024
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from constants import (
    CAT_COLUMNS,
    QUANT_COLUMNS,
    ATTRITION_COL_NAME,
    EDA_IMGS_PTH,
    RES_IMGS_PTH,
    MODELS_PTH,
    BANK_DATA_PTH,
    FEAT_IMP_PLOT_NAME,
    LR_SCORE_PLOT_NAME,
    RFC_SCORE_PLOT_NAME,
    ROC_PLOT_NAME)

sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def create_distribution_plot(
        df, output_pth, columns_names, is_categorical=True):
    """
        creates distribution plots for the given columns of the df Datafrage
        input:
                df: pandas dataframe
                output_pth: path where to save the plots
                columns_names: list with names of the columns to use
                is_categorical: bool, represents if the columns have categorical
                                or quantitative values
        output:
                None
        """
    for col_name in columns_names:
        figure = plt.figure(figsize=(20, 15))
        if is_categorical:
            df[col_name].value_counts("normalize").plot(kind="bar")
        else:
            sns.histplot(df[col_name], stat="density", kde=True)
        filename = f"{output_pth}{col_name} distribution.png"
        figure.savefig(filename)
        plt.close(figure)


def perform_eda(df, output_pth):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            output_pth: path where to save the plots

    output:
            None
    """
    # performing categorical plot
    create_distribution_plot(
        df,
        output_pth,
        CAT_COLUMNS +
        [ATTRITION_COL_NAME],
        is_categorical=True)

    # performing quantitative plots
    create_distribution_plot(
        df,
        output_pth,
        QUANT_COLUMNS,
        is_categorical=False)

    # correlation plot
    figure = plt.figure(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    filename = (f"{output_pth}correlation matrix.png")
    figure.savefig(filename)


def encoder_helper(df, category_lst):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            (df, one_hot_category_lst): tuple
                df: pandas dataframe with new columns for
                one_hot_category_lst: list of names for the categorical features
                                      with one hot encoding
    """
    df_one_hot = pd.get_dummies(df, columns=category_lst)
    one_hot_category_lst = []
    for cat_col in category_lst:
        one_hot_category_lst.extend(
            [f"{cat_col}_{val_name}" for val_name in df[cat_col].unique()])
    return df_one_hot, one_hot_category_lst


def perform_feature_engineering(df):
    """
    input:
              df: pandas dataframe
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    df['Churn'] = df[ATTRITION_COL_NAME].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df, one_hot_cat_columns = encoder_helper(df, CAT_COLUMNS)
    keep_cols = QUANT_COLUMNS + one_hot_cat_columns

    y = df['Churn']
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def create_scores_report_figure(
        y_train, y_test, y_train_preds, y_test_preds, model_type_name):
    """
        creates a figure with the score results
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds: training predictions
                y_test_preds: test predictions
                model_type_name: the name of the model used for the predictions

        output:
                 figure: matplotlib Figure, figure with the scores results
    """
    # train results
    figure = plt.figure(figsize=(7, 5))
    plt.text(0.01, 0.9, str(f'{model_type_name} Train:'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.5, str(classification_report(y_train, y_train_preds)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!

    # Test results
    plt.text(0.01, 0.45, str(f'{model_type_name} Test:'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!

    plt.axis('off')
    return figure


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_pth):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            output_pth: path to store the figure

    output:
             None
    """

    rf_report_figure = create_scores_report_figure(
        y_train, y_test, y_train_preds_rf, y_test_preds_rf, "Random Forest")

    rf_report_figure.savefig(
        f"{output_pth}{RFC_SCORE_PLOT_NAME}")
    lr_report_figure = create_scores_report_figure(
        y_train, y_test, y_train_preds_lr, y_test_preds_lr, "Logistic Regression")
    lr_report_figure.savefig(
        f"{output_pth}{LR_SCORE_PLOT_NAME}")


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    figure = plt.figure(figsize=(20, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    figure.savefig(f"{output_pth}{FEAT_IMP_PLOT_NAME}")


def roc_plot(lr_model, rfc_model, X_test, y_test, output_pth):
    """
    creates and stores the roc curve plot in output_pth
    input:
            lr_model: model object of logistic regression
            rfc_model: model of Random Forest regression
            X_test: X testing data
            y_test: y testing data
            output_pth: path to store the figure

    output:
             None
    """
    lr_plot = plot_roc_curve(lr_model, X_test, y_test)
    figure = plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(rfc_model, X_test, y_test, ax=axis, alpha=0.8)
    lr_plot.plot(ax=axis, alpha=0.8)
    figure.savefig(f"{output_pth}{ROC_PLOT_NAME}")


def train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        models_out_pth,
        res_out_pth):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """

    # training of models
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # logistic regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)

    # get predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # create score images

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
        output_pth=res_out_pth)

    # create ROC and feature importance plot
    roc_plot(
        lrc,
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        output_pth=res_out_pth)
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_test,
        output_pth=res_out_pth)

    # saving the models
    joblib.dump(cv_rfc.best_estimator_, f'{models_out_pth}rfc_model.pkl')
    joblib.dump(lrc, f'{models_out_pth}logistic_model.pkl')


if __name__ == "__main__":
    bank_df = import_data(BANK_DATA_PTH)
    perform_eda(bank_df, EDA_IMGS_PTH)
    X_train_set, X_test_set, y_train_set, y_test_set = perform_feature_engineering(
        bank_df)
    train_models(
        X_train_set,
        X_test_set,
        y_train_set,
        y_test_set,
        MODELS_PTH,
        RES_IMGS_PTH)
