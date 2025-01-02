import numpy as np

"""
    Implementing softmax regression from scratch
"""

class softmax_regression():
    
    def __init__(self, k=2, learning_rate=0.01, n_iterations=1000):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_classes = k
        
    def softmax(self, z):
        # compute softmax values for each set of scores
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        n_samples, n_features = X_train.shape
        
        # one-hot encode y_train
        y_encoded = np.eye(self.n_classes)[y_train]
        
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)

        # gradient Descent
        for _ in range(self.n_iterations):
            # Compute scores and probabilities
            logits = np.dot(X_train, self.weights) + self.bias
            y_pred = self.softmax(logits)

            # compute gradients
            gradient_w = (1 / n_samples) * np.dot(X_train.T, (y_pred - y_encoded))
            gradient_b = (1 / n_samples) * np.sum(y_pred - y_encoded, axis=0)

            # update weights and biases
            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

    
    def calc_prob(self, X):
        
        logits = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(logits)
        return y_pred
        
    def predict(self, X_test):
        logits = np.dot(X_test, self.weights) + self.bias
        y_pred = self.softmax(logits)
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, y_test):
        pass