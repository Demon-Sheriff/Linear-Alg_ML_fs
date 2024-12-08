import numpy as np

""" we can maintain each unique word in a trie and also store the frequency of each word class_wise [LATER]"""

class NaiveBayesSpamClassifier():
    
    def __init__(self):
        self.class_priors = None 
        self.feature_likelihoods = None
        self.n_classes = None
        self.word_dict = {}
    
    def fit(self, X_train, y_train):
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.n_classes = len(np.unique(y_train)) # number of classes is the number of unique labels in y_train.

        # calculate the class_priors
        """ d is the length of the vocabulary (which is basically the total number of unique words in our traning sample) """
        n_samples, d = X_train.shape 
        
        self.class_priors = np.zeros(self.n_classes) # intialise the class priors array index 0 is the start of the class. classes -> 0, 1, 2, 3, .... c
        class_count = np.zeros(self.n_classes)
        
        # accumulate the counts of each class/label
        for label in y_train:
            self.class_priors[label] += 1
            
        for idx,count in enumerate(self.class_priors):
            class_count[idx] = count
            self.class_priors[idx] /= n_samples
            
        # calculate the feature likelihoods for each class
        self.feature_likelihoods = np.zeros((d, self.n_classes))
        
        for class_ in range(self.n_classes):
            class_indices = np.where(y_train == class_)[0]

            self.feature_likelihoods[:, label] = np.sum(X_train[class_indices], axis=0)
            self.feature_likelihoods[:, label] = self.feature_likelihoods[:, label] / class_count[label]