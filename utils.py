"""
UTIL SCRIPT
Oisin Mc Laughlin
22441106
"""

import pandas as pd
import matplotlib.pyplot as plt

# Reading train and test data
train_data = pd.read_csv('wildfires_training.csv')
test_data = pd.read_csv('wildfires_test.csv')

def get_data():
    """
    Utility function to read in the wildfire training and test data from CSV files,
    separate features and target variable and return them for model training and evaluation.

    Returns:
        x_train, y_train, x_test, y_test
    """
    # Read the training and test data csv file with pandas

    print("\nTraining Data:")
    print(train_data.info())

    print("\nTest Data:")
    print(test_data.info())

    # Separate features and target data
    x_train = train_data.drop('fire', axis=1)
    y_train = train_data['fire']
    x_test = test_data.drop('fire', axis=1)
    y_test = test_data['fire']

    print("\nFeatures and targets seperated:")
    print(x_train.head())
    print(y_train.head())

    return x_train, y_train, x_test, y_test


def visualise_data():
    """
    Utility function to visualise the training and test data using histograms.
    Returns:
        None
    """
    # Exclude day, month, year cols
    visual_train_data = train_data.drop(['day', 'month', 'year'], axis=1)
    visual_test_data = test_data.drop(['day', 'month', 'year'], axis=1)

    # Visualise training and test data with histograms
    visual_train_data.hist(figsize = (12, 10))
    plt.suptitle("Training Data")
    plt.tight_layout()
    plt.show()

    visual_test_data.hist(figsize = (12, 10))
    plt.suptitle("Test Data")
    plt.tight_layout()
    plt.show()
