"""
LOGISTIC REGRESSION CLASSIFIER ON WILDFIRE DATASET
Oisin Mc Laughlin
22441106
"""

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from utils import get_data

# Get the data using the utility function
x_train, y_train, x_test, y_test = get_data()

# Init logistic regression model with max iterations of 1000
lr_default = LogisticRegression(max_iter=1000)
lr_default.fit(x_train, y_train)

# Testing how well the model performs on the both the training and test data
train_predictions = lr_default.predict(x_train)
train_accuracy = metrics.accuracy_score(y_train, train_predictions)

test_predictions = lr_default.predict(x_test)
test_accuracy = metrics.accuracy_score(y_test, test_predictions)

print("\nTesting linear regression model with default parameters")
print("Training Accuracy: ", train_accuracy * 100, "%")
print("Test Accuracy: ", test_accuracy * 100, "%")


# Hyperparam tuning
penalty = [None , 'l2', 'l1', 'elasticnet']
tol = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

print("\nChoosing both penalty and tol hyperparameters to tune based on these values:")
print("Penalty: ", penalty)
print("Tol: ", tol)
print("Default penalty is l2 and default tol is 0.0001")

results = []
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


        # Testing how well the model performs on the training and test data
        training_predictions = lr_tuned.predict(x_train)
        training_accuracy = metrics.accuracy_score(y_train, training_predictions)

        test_predictions = lr_tuned.predict(x_test)
        test_accuracy = metrics.accuracy_score(y_test, test_predictions)

        # Record the results
        results.append({
            'penalty': p,
            'tol': t,
            'train_accuracy': training_accuracy,
            'test_accuracy': test_accuracy
        })

        # If new best accuracy, store the parameters
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_params = {'penalty': p, 'tol': t}

print("\nBest Accuracy: ", best_accuracy * 100, "%")
print("Best Parameters: ", best_params)


# Plot results
for p in penalty:
    tol_values = []
    accuracy_values = []

    # For each result, if the penalty matches, add the tol and accuracy to the lists
    for r in results:
        if r['penalty'] == p:
            tol_values.append(r['tol'])
            accuracy_values.append(r['test_accuracy'] * 100)

    # Plot the values for this penalty, with larger points and some transparency
    plt.scatter(tol_values, accuracy_values, label="Penalty: " + str(p), s = 100, alpha = 0.7)
    plt.plot(tol_values, accuracy_values)

# Labels and title etc
plt.xlabel("Tol")
plt.ylabel("Accuracy (%)")
plt.title("Logistic Regression Hyperparameter Tuning")
plt.legend()
plt.tight_layout()
plt.xscale('log')

plt.show()
