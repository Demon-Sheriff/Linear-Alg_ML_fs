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
        return [1 if value > self.threshold else 0 for value in z]
        
    def fit(self, X_train, y_train):
        
        n_samples, n_features = X_train.shape
        # start off with random weights and biases
        self.weights = np.zeros(n_features)
        self.bias = 0 # start off with zero as the bias
        
        for _ in self.n_iterations:
            # apply the weighted sum or the dot product of theta.T X [Calculating the perception]
            z = np.dot(self.weights, X_train) + self.bias 
            
            # process z to the activation function [basically make predictions based on the perception]
            z_predictions = self.activation_fn(z=z)
            
            # update the weights and baises