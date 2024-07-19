# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project we implement a library for customer churn prediction using clean code principles. For this code we use 
a dataset from a bank.
This project has code for the following tasks:
- Data exploration Analysis (EDA) and the creation of the respective plots
- Feature Engineering.
- Training models using models of the scikit-learn library
- Evaluating these models and creating the evaluation plots

## Requirements
- python 3.6 (we tested the code with this version)
- modules in requirements_py3.6.txt (install with `pip install -r requirements_py3.6.txt`)

## Files and data description
Overview of the files and data present in the root directory. 

#### Folders
- data: directory with bank data
- images:
  - eda: directory with the plots of EDA
  - results: directory with roc curve plot and scores of the evaluation
- logs: directory with the lof of the test script
- models: saved models after training

#### Files
- churn_library.py: module performing prediction, training and creating plots
- constants.py: file with constants used in production code and testing
- churn_script_logging_and_tests.py: tests (for pytest) of the churn_library module
- conftest.py: configuration of the tests
- pytest.ini: configuration of pytest (configuration of the log file)
- churn_notebook.ipynb: jupyter notebook with initial code for the implementation of the module
- Guide.ipynb: jupyter notebook with instructions for this project of Udacity
- requirements_py3.6.txt: dependencies of the project


## Running Files
How do you run your files? What should happen when you run your files?

#### Creating plots and training the model
Run

`python churn_library.py`


#### Running the tests
Run

`pytest churn_script_logging_and_tests.py`

Notes:
- The configuration for the log file of the tests is in the file pytest.ini
- The test create temporal folders images/eda_test/, images/results_test/ and models_test/ to save the results during
testing. When finished, these folders are deleted automatically.




