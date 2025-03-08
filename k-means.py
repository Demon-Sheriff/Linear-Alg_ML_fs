import numpy as np
class KMeans():
  def __init__(self, n_clusters, tol = 1e-4, max_iter=300, init_type='k-means++', custom_init=None) -> None:
    self.cluster_centroids = None
    self.n_clusters = n_clusters
    self.init_type = init_type
    self.custom_centroids = custom_init # if the user wants a custom init of the centroids
    self.tol = tol
    self.max_iter = max_iter

  def random_init(self, X, k):
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

  def kmeans_plus_plus_init(self, X, k):
    np.random.seed(42)  
    n_samples = X.shape[0]
    
    # choose first centroid randomly
    centroids = [X[np.random.choice(n_samples)]]
    
    for _ in range(1, k):
        # compute distances of all points to the nearest centroid
        distances = np.min([np.linalg.norm(X - c, axis=1)**2 for c in centroids], axis=0)
        # choose the next centroid with probability proportional to distance²
        probabilities = distances / np.sum(distances)
        next_centroid = X[np.random.choice(n_samples, p=probabilities)]
        centroids.append(next_centroid)
    return np.array(centroids)

  def fit(self, X):

    X = np.array(X)
    k = self.n_clusters
    n_samples, n_features = X.shape

    # init the centroids
    if self.init_type == 'random':
      self.cluster_centroids = self.random_init(X, k)
    elif self.init_type == 'custom':
      self.cluster_centroids = self.custom_centroids
    else:
      self.cluster_centroids = self.kmeans_plus_plus_init(X, k)

    centroids = np.array(self.cluster_centroids)

    for _ in range(self.max_iter):
      # compute distances of each point from the centroids and find the min dist centroid then assign centroids to each point
      assignments = np.argmin([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0) # (n_samples, 1) going with euclidean distance for now...
      new_centroids = np.copy(centroids)
      # update the centroids
      for c in range(k):
          cluster_points = X[assignments == c]
          if len(cluster_points) > 0:
              new_centroids[c] = np.mean(cluster_points, axis=0)
          else: # handle empty clusters...
              new_centroids[c] = X[np.random.choice(n_samples)]

      # check for convergence based on tolerance
      shift = np.linalg.norm(new_centroids - centroids, ord='fro')
      if shift < self.tol:
          break  

      centroids = new_centroids
      
    self.cluster_centroids = centroids

  def predict(self, X):

    centroids = self.cluster_centroids
    if centroids is None:
      print('please run fit first')
      return
    
    assignments = np.argmin([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
    return assignments

  def fit_predict(self, X):
    self.fit(X)
    return self.predict(X)
  