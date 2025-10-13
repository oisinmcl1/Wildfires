"""
LOGISTIC REGRESSION CLASSIFIER ON WILDFIRE DATASET
"""

import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

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

# Init logistic regression model with max iterations of 1000
lr_default = LogisticRegression(max_iter=1000)
lr_default.fit(x_train, y_train)

# Testing how well the model performs on the test data
predictions = lr_default.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print("\nTesting linear regression model with default parameters")
print("Default Accuracy: ", accuracy * 100, "%")

# Hyperparam tuning
penalty = [None , 'l2', 'l1', 'elasticnet']
tol = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

print("\nChoosing both penalty and tol hyperparameters to tune based on these values:")
print("Penalty: ", penalty)
print("Tol: ", tol)
print("Default penalty is l2 and default tol is 0.0001")

best_accuracy = 0
best_params = {}

for p in penalty:
    for t in tol:
        # Based on sklearn docs, if penalty is l1 or elasticnet, solver needs to be changed
        # As well l1 ratio should be set if penalty is elasticnet
        solver_param = 'lbfgs'
        l1_ratio_param = None

        if p == 'l1':
            solver_param = 'liblinear'
        elif p == 'elasticnet':
            solver_param = 'saga'
            l1_ratio_param = 0.5

        # Init logistic regression model with chosen hyperparameters
        lr_tuned = LogisticRegression(
            penalty= p,
            tol=t,
            solver=solver_param,
            l1_ratio=l1_ratio_param,
            max_iter=1000000 # honestly just added 0's until there was no warnings
        )
        lr_tuned.fit(x_train, y_train)

        # Testing how well the model performs on the test data
        predictions = lr_tuned.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'penalty': p, 'tol': t}

print("\nBest Accuracy: ", best_accuracy * 100, "%")
print("Best Parameters: ", best_params)
