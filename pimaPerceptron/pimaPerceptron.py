import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron:
    def __init__(self, eta=0.1, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Load Dataset
df = pd.read_csv('pima-indians-diabetes.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = np.where(y == 0, -1, 1)

# Define a range of learning rates and iterations to experiment with
learning_rates = [0.01, 0.05, 0.1, 0.5, 1]
n_iters = 50
best_accuracy = 0
best_perceptron = None

best_accuracy = 0
best_lr = 0
best_iter = 0
best_classifier = None

# Find the best learning rate
for eta in learning_rates:
    perceptron = Perceptron(eta=eta, n_iter=n_iters, random_state=1)
    perceptron.fit(X, y)
    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_perceptron = perceptron

# Print the best learning rate and its accuracy
print(f'Best learning rate: {best_perceptron.eta}, Accuracy: {best_accuracy}')

# Plot the misclassification errors against the number of epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(best_perceptron.errors_) + 1), best_perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title(f'Misclassifications vs. Epochs (Learning Rate: {best_perceptron.eta})')
plt.tight_layout()
plt.show()

