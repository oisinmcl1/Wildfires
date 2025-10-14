"""
SUPPORT VECTOR MACHINE CLASSIFIER FOR WILDFIRE PREDICTION
Oisin Mc Laughlin
22441106
"""

from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from utils import get_data

# Get the data using the utility function
x_train, y_train, x_test, y_test = get_data()

# Init SVM model
svm_default = SVC()
svm_default.fit(x_train, y_train)

# Testing how well the model performs on the both the training and test data
train_predictions = svm_default.predict(x_train)
train_accuracy = metrics.accuracy_score(y_train, train_predictions)

test_predictions = svm_default.predict(x_test)
test_accuracy = metrics.accuracy_score(y_test, test_predictions)

print("\nTesting SVM model with default parameters")
print("\nTraining Accuracy: ", train_accuracy * 100, "%")
print("Test Accuracy: ", test_accuracy * 100, "%")


# Hyperparam tuning
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
c_vals = [0.1, 1, 10, 100, 1000]

print("\nChoosing both kernel and C hyperparameters to tune based on these values:")
print("Kernel: ", kernel)
print("C: ", c_vals)
print("Default kernel is rbf and default C is 1.0")

results = []
best_accuracy = 0
best_params = {}

for k in kernel:
    for c in c_vals:
        # Init SVM model with chosen hyperparameters
        svm_tuned = SVC(
            kernel=k,
            C=c
        )
        svm_tuned.fit(x_train, y_train)

        # Testing how well the model performs on the training and test data
        train_predictions = svm_tuned.predict(x_train)
        train_accuracy = metrics.accuracy_score(y_train, train_predictions)

        test_predictions = svm_tuned.predict(x_test)
        test_accuracy = metrics.accuracy_score(y_test, test_predictions)

        results.append({
            'kernel': k,
            'c': c,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        })

        # If new best accuracy, store the parameters
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_params = {'kernel': k, 'C': c}

print("\nBest accuracy: ", best_accuracy * 100, "%")
print("Best parameters: ", best_params)


# Plot results
for k in kernel:
    c_values = []
    accuracy_values = []

    # For each result, if the kernel matches, add the c and accuracy to the lists
    for r in results:
        if r['kernel'] == k:
            c_values.append(r['c'])
            accuracy_values.append(r['test_accuracy'] * 100)

    # Plot the values for this kernel, with larger points and some transparency
    plt.scatter(c_values, accuracy_values, label= "Kernal: " + k, s = 100, alpha = 0.7)
    plt.plot(c_values, accuracy_values)

# Labels and title etc
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.title("SVM Hyperparameter Tuning")
plt.legend()
plt.tight_layout()

plt.show()