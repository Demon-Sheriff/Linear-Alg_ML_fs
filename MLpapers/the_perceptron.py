# Implementing the perceptron paper 

"""
    Source of the paper : 
    https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf

    Dated year 1958, University of Pensylvannia 
    Written by F Rosenblatt

"""

""" 
    NOTE: 
        Draw-backs of the perceptron : 
        1. Does not work on datasets where the data is not linearly seperable
        2. Assumes that the decision boundary will always be linear on the training data and while making predictions.
"""

import numpy as np

class Perceptron():
    
    def __init__(self, threshold, learning_rate=0.001, n_iterations=1000):
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations 
        self.weights = None
        self.bias = None
        self.threshold = threshold
        
    def activation_fn(self, z):
        return np.where(z > self.threshold, 1, 0)
        
    def fit(self, X_train, y_train):
        
        n_samples, n_features = X_train.shape
        # start off with random weights and biases
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0 # start off with zero as the bias
        
        # TODO : add convergence check in the update rule.
        
        """ 
            IMPORTANT NOTE: 

            The perceptron typically updates weights one sample at a time (stochastic gradient descent). 
            We need to loop over each sample individually within the iteration loop.
        """
        
        for _ in range(self.n_iterations):
            
            for idx, x_i in enumerate(X_train):
                
                # apply the weighted sum or the dot product of theta.T X [Calculating the perception]
                z = np.dot(x_i, self.weights) + self.bias 
                
                # process z to the activation function [basically make predictions based on the perception]
                y_hat = self.activation_fn(z)
                
                # update the weights and baises
                if y_hat != y_train[idx]:
                    self.weights = self.weights + self.learning_rate * np.dot(x_i, (y_train[idx] - y_hat))
                    self.bias = self.bias + self.learning_rate * np.sum(y_train[idx] - y_hat)
            
        return self.weights, self.bias
    
    def predict(self, X_test):
        
        X_test = np.array(X_test) # make sure X_test is a numpy array
        z_test = np.dot(X_test, self.weights) + self.bias
        y_pred = self.activation_fn(z_test)
        
        return np.array(y_pred)
    
    def evaluate():
        # TODO : add evaluation metrics
        pass
    
    