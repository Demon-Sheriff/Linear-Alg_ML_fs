import numpy as np

class NaiveBayesSpamClassifier():
    
    def __init__(self):
        self.class_priors = None 
        self.feature_likelihoods = None
        pass
    
    def fit(self, X_train, y_train):
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # calculate the class_priors
        """ d is the length of the vocabulary (which is basically the total number of unique words in our traning sample) """
        n_samples, d = X_train.shape 
        
        self.class_priors = []*d # intialise the class priors array index 0 is the start of the class. classes -> 0, 1, 2, 3, .... c
        for label in y_train:
            self.class_priors[label] += 1
            
        for class_prior in self.class_priors:
            class_prior /= n_samples
            
        # calculate the feature likelihoods for each class
        pass