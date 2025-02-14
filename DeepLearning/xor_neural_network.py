import numpy as np

class Network:

    def __init__(self, input_size, hidden_size, output_size, activation_fn ='sigmoid', n_iters=1000, lr=0.01):

        self.hidden_size = hidden_size
        self.n_iters = n_iters
        self.input_size = input_size
        self.output_size = output_size
        self.activation_type = activation_fn
        self.learning_rate = lr
        self.weights = None
        self.biases = None

    def activation_function(self, z):
        
        """
        Applies the specified activation function to the input z.

        Args:
        - activation_type (str): The type of activation function ('sigmoid', 'relu', 'tanh', 'softmax').
        - z (numpy.ndarray): The input value to be activated.

        Returns:
        - Activated output (numpy.ndarray): The activated value after applying the selected activation function.
        """
        
        if self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))  # sigmoid activation
            
        elif self.activation_type == 'relu':
            return np.maximum(0, z)  # ReLU activation
            
        elif self.activation_type == 'tanh':
            return np.tanh(z)  # tanh activation
            
        elif self.activation_type == 'softmax':
            # softmax activation
            exp_z = np.exp(z - np.max(z))  # To prevent overflow
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        
        else:
            raise ValueError("unsupported activation function type: {}".format(self.activation_type))

    def activation_derivative(self, a):

        if self.activation_type == 'sigmoid':
            return a * (1 - a)
        elif self.activation_type == 'tanh':
            return 1 - a ** 2
        elif self.activation_type == 'relu':
            return (a > 0).astype(float)
        else:
            raise ValueError("Unsupported activation function.")

    def xavier_init(self, shape):
        
        """
        Xavier/Glorot weight initialization for Sigmoid.
        
        Args:
        - shape (tuple): The shape of the weight matrix (e.g., (n_in, n_out)).
        
        Returns:
        - weights (numpy.ndarray): Initialized weight matrix.
        """
        # xavier/glorot initialization formula: uniform distribution
        n_in, n_out = shape
        limit = np.sqrt(6 / (n_in + n_out))  # calculate the limit for uniform distribution
        wts = np.random.uniform(-limit, limit, size=shape)  # uniform distribution within [-limit, limit]
        return wts

    def train(self, X_train, y_train):

        n_features, n_samples = X_train.shape # n_features, m_examples = num_samples

        # initialise w[l]s and b[l]s
        w1 = self.xavier_init((self.hidden_size, n_features))
        b1 = np.zeros((self.hidden_size, 1))
        w2 = self.xavier_init((self.output_size, self.hidden_size))
        b2 = np.zeros((self.output_size, 1))

        # apply gradient descent 
        for _ in range(self.n_iters):

            # forward pass
            z1 = np.dot(w1, X_train) + b1
            a1 = self.activation_function(z1)
            z2 = np.dot(w2, a1) + b2
            a2 = self.activation_function(z2)

            # backward pass
            dz2 = a2 - y_train
            del_w2 = np.dot(dz2, a1.T) / n_samples
            del_b2 = np.sum(dz2, axis=1, keepdims=True) / n_samples

            dz1 = np.dot(w2.T, dz2) * self.activation_derivative(a1)
            del_w1 = np.dot(dz1, X_train.T) / n_samples
            del_b1 = np.sum(dz1, axis=1, keepdims=True) / n_samples

            # del_b2 = np.dot((a2 - y_train), np.ones((n_samples, n_samples)).T) / n_samples
            # del_b1 = np.dot(w2.T, (a2 - y_train)) * (a1 * (1 - a1)) / n_samples
            # del_w2 = np.dot((a2 - y_train), a1.T) / n_samples
            # del_w1 = np.dot(np.dot(w2.T, (a2 - y_train)) * (a1 * (1 - a1)), X_train.T) / n_samples

            # update the weigts and biases
            w1 -= self.learning_rate*del_w1
            w2 -= self.learning_rate*del_w2
            b1 -= self.learning_rate*del_b1
            b2 -= self.learning_rate*del_b2

        self.weights = {
            'w1': w1,
            'w2': w2,
        }
        self.biases = {
            'b1': b1,
            'b2': b2,
        }
    
    def predict(self, X_test):

        # extract the biases
        b1 = self.biases['b1']
        b2 = self.biases['b2']

        # extract the weights
        w1 = self.weights['w1']
        w2 = self.weights['w2']

        # final forward pass
        z1 = np.dot(w1, X_test) + b1
        a1 = self.activation_function(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = self.activation_function(z2)

        y_hat = a2

        return (y_hat > 0.5).astype(int)