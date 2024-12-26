import numpy as np

def r2_score(y_test, y_pred):
    
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    
    y_mean = np.mean(y_test)
    
    ss_residual = np.sum((y_pred - y_test) ** 2)
    ss_mean = np.sum((y_test - y_mean) ** 2)
    
    return (1 - (ss_residual/ss_mean))

def linear_reg(X_train, y_train, n_iterations = 1000, learning_rate = 0.01):
    
    n, m = X_train.shape # n_samples, m_features
    w = np.zeros(m)
    b = 0
    
    # applying batch gradient descent.
    for _ in range(n_iterations):
        
        z = np.dot(X_train, w) + b
        dw = (1 / n) * np.dot(X_train.T, (z - y_train))
        db = (1 / n) * np.sum(z - y_train)
        
        w -= learning_rate * dw
        b -= learning_rate * db
    
    return w,b
