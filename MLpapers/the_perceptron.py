# Implementing the perceptron paper 

"""
Source of the paper : 
https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf

Dated year 1958, University of Pensylvannia 
Written by F Rosenblatt

"""

import numpy as np
import pandas as pd

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
        
        for _ in range(self.n_iterations):
            # apply the weighted sum or the dot product of theta.T X [Calculating the perception]
            z = np.dot(self.weights, X_train) + self.bias 
            
            # process z to the activation function [basically make predictions based on the perception]
            y_hat = self.activation_fn(z)
            
            # update the weights and baises
            if y_hat != y_train:
                self.weights = self.weights + self.learning_rate * np.dot(X_train.T, (y_train - y_hat))
                self.bias = self.bias + self.learning_rate * np.sum(y_train - y_hat)
            
        return self.weights, self.bias
    
    def predict():
        # To implement
        pass