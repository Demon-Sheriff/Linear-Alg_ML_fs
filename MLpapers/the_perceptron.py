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
    
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations 
        