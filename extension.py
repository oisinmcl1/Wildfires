"""
PLACEHOLDER
"""

import pandas as pd

# Read the training and test data csv file with pandas
train_data = pd.read_csv('wildfires_training.csv')
test_data = pd.read_csv('wildfires_test.csv')

print("Reading in the data:")
print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
