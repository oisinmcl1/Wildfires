"""
PLACEHOLDER
"""

import pandas as pd

# Read the training and test data csv file with pandas
train_data = pd.read_csv('wildfires_training.csv')
test_data = pd.read_csv('wildfires_test.csv')

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
