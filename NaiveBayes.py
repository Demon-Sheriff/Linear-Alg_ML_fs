import numpy as np

""" we can maintain each unique word in a trie and also store the frequency of each word class_wise [LATER]
    TODO:
        1. make all the array operations vectorized
        2. implement trie for word processing
"""

class TrieNode:
    def __init__(self):
        self.children = {}
        self.class_counts = [0, 0]  # For binary classification: Spam (1) and Ham (0)

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, label):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.class_counts[label] += 1

    def get_class_counts(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return [0, 0]  # Word not found
            node = node.children[char]
        return node.class_counts

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
            # find all rows corresponding to the current class
            class_indices = np.where(y_train == class_)[0]

            # sum up word frequencies for the current class
            self.feature_likelihoods[:, class_] = np.sum(X_train[class_indices], axis=0)

            # normalize by the total samples in the class
            self.feature_likelihoods[:, class_] /= class_count[class_]
            
        return self
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        n_samples, d = X_test.shape
        y_pred = []
        
        for i in range(n_samples):
            
            log_probs = []
            for label in range(self.n_classes):
                log_prior = np.log(self.class_priors[label])
                log_likelihood = np.sum(
                    X_test[i] * np.log(self.feature_likelihoods[:, label]) +
                    (1 - X_test[i]) * np.log(1 - self.feature_likelihoods[:, label])
                )
                log_probs.append(log_prior + log_likelihood)
            
            y_pred.append(np.argmax(log_probs))
        
        return y_pred