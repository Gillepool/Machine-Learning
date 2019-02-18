import numpy as np 

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X-mu)/sigma, mu, sigma

#Principal component analysis.
def pca(X):
    m = X.shape[0]

    sigma = X.T.dot(X)/m
    U, S, _ = np.linalg.svd(sigma, full_matrices=True)

    return U, S

def compute_centroids(X, idx, K):
    n = X.shape[1]

    centroids = np.zeros((K, n))
    num = np.zeros((K, 1))
    sum = np.zeros((K, n))

    for i in range(idx.shape[0]):
        z = idx[i]
        num[z] += 1
        sum[z, :] += X[i, :]
    centroids = sum / num

    return centroids

def find_closests_centroids(X, centroids):
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min = np.inf
        for j in range(K):
            diff = np.sum(np.power(X[i, :] - centroids[j, : ], 2))
            if min > diff:
                min = diff
                idx[i] = j
    idx = idx.astype(int)
    return idx

def init_centroids(X, K):
    rand_idx = np.random.permutation(X.shape[0])
    centroids = X[rand_idx[0:K], :]
    return centroids

#K-means
def Kmeans(X, init_centroid, max_iters, plot_progress=None):
    m = X.shape[0]
    K = init_centroid.shape[0]

    centroids = init_centroid
    #prev_centroids = centroids

    idx = np.zeros(m)
    idx = idx.astype(int)

    for _ in range(max_iters):
        idx = find_closests_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    
    return centroids, idx

def project_data(X, U, K):
    m = X.shape[0]
    Z = np.zeros((m, K))
    for i in range(m):
        x = X[i, :]
        Z[i, :] = np.dot(x, U[:, 0:K])
    return Z


def recover_data(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    for i in range(Z.shape[0]):
        v = Z[i, :]
        for j in range(U.shape[0]):
            X_rec[i, j] = np.dot(v, U[j, 0:K].T)
    return X_rec