"""
constants for churn library and respective tests

author: Erick Alvarez
date: July 19 2024
"""
BANK_DATA_PTH = "./data/bank_data.csv"

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

ATTRITION_COL_NAME = 'Attrition_Flag'

ALL_COL_NAMES = CAT_COLUMNS + QUANT_COLUMNS + [ATTRITION_COL_NAME]

EDA_IMGS_PTH = "./images/eda/"
RES_IMGS_PTH = "./images/results/"
MODELS_PTH = "./models/"

FEAT_IMP_PLOT_NAME = "feature_importances.png"
LR_SCORE_PLOT_NAME = "logistic_regression_classification_scores.png"
RFC_SCORE_PLOT_NAME = "random_forest_classification_scores.png"
ROC_PLOT_NAME = "roc_curve.png"

RESULTS_PLOTS_NAMES = [
    FEAT_IMP_PLOT_NAME,
    LR_SCORE_PLOT_NAME,
    RFC_SCORE_PLOT_NAME,
    ROC_PLOT_NAME
]

# Testing Constants

EDA_IMGS_TEST_PTH = "./images/eda_test/"
RES_IMGS_TEST_PTH = "./images/results_test/"
MODELS_TEST_PTH = "./models_test/"

TEST_DATA = [
    ['M', 'Uneducated', 'Married', '40K-60K', "Blue",
     50, 2, 30, 2, 5, 6, 2000, 1500, 16000, 0.8, 5000, 70, 0.2, 0.3, "Attrited Customer"],
    ['F', 'Doctorate', 'Married', 'Less than $40K', "Silver",
     42, 2, 30, 2, 5, 6, 4000, 1500, 15000, 0.4, 4000, 70, 0.2, 0.2, "Existing Customer"],
    ['F', 'Graduate', 'Single', '40K-60K', "Blue",
     20, 2, 30, 2, 5, 6, 7000, 1500, 16000, 0.2, 4000, 70, 0.7, 0.5, "Attrited Customer"],
    ['M', 'College', 'Divorced', 'Less than $40K', "Silver",
     36, 2, 30, 2, 5, 6, 8000, 1500, 10000, 0.8, 4000, 70, 0.5, 0.7, "Existing Customer"],
    ['F', 'Graduate', 'Married', '40K-60K', "Blue",
     25, 2, 30, 2, 9, 6, 9000, 1500, 28000, 0.8, 4000, 70, 0.4, 0.4, "Existing Customer"],
    ['M', 'High School', 'Married', 'Less than $40K', "Silver",
     40, 2, 30, 2, 8, 6, 7000, 1500, 2000, 0.8, 4000, 70, 0.2, 0.2, "Existing Customer"],
    ['F', 'Graduate', 'Single', '40K-60K', "Blue",
     20, 2, 30, 2, 5, 6, 7000, 1500, 16000, 0.2, 4000, 70, 0.7, 0.5, "Existing Customer"],
    ['M', 'College', 'Single', 'Less than $40K', "Blue",
     36, 2, 30, 2, 2, 6, 8000, 1600, 10000, 0.8, 4000, 70, 0.5, 0.7, "Existing Customer"],
    ['F', 'Graduate', 'Married', '40K-60K', "Blue",
     25, 2, 30, 2, 4, 6, 9000, 2000, 28000, 0.8, 4000, 70, 0.4, 0.4, "Existing Customer"],
    ['M', 'High School', 'Married', 'Less than $40K', "Silver",
     40, 2, 30, 2, 5, 6, 7000, 1500, 2000, 0.8, 4000, 70, 0.2, 0.2, "Attrited Customer"]
]

TEST_DATA = TEST_DATA * 4  # just to make set big enough for training

CATEGORY_VALS_DICT = {
    'Gender': [
        'M',
        'F'],
    'Education_Level': [
        'Uneducated',
        'Doctorate',
        'Graduate',
        'College',
        'High School'],
    'Marital_Status': [
        'Married',
        'Single',
        'Divorced'],
    'Income_Category': [
        '40K-60K',
        'Less than $40K'],
    'Card_Category': [
        "Blue",
        "Silver"]}
