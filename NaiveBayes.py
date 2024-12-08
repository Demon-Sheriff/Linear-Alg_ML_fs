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
        
        self.class_priors = []*self.n_classes # intialise the class priors array index 0 is the start of the class. classes -> 0, 1, 2, 3, .... c
        class_count = []*self.n_classes
        
        # accumulate the counts of each class/label
        for label in y_train:
            self.class_priors[label] += 1
            
        for idx,class_prior in enumerate(self.class_priors):
            class_count[idx] = class_prior
            class_prior /= n_samples
            
        # calculate the feature likelihoods for each class
        self.feature_likelihoods = [[] for _ in range(self.n_classes)]*d
        
        # for xij in range(d):
        #     for class_ in range(self.n_classes):
        #         # frequency of xij for the current class
                
        #         idxs = []
                
        #         freq_xij = 
        #         self.feature_likelihoods[xij][class_] = 1/class_count[class_]
        
        for class_ in range(self.n_classes):
            for word in range(d):
                for i in range(n_samples):
                    self.feature_likelihoods[word][class_] += X_train[i][word] * (y_train[i] == class_)
                
            self.feature_likelihoods[word][class_] /= class_count[class_]