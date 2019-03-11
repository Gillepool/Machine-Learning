import numpy as np

def affine_forward(X, w, b):
    m = X.shape[0]
    out = X.reshape(m , -1).dot(w) + b
    cache = (X, w, b)
    return out, cache

def affine_backward(dOut, cache):
    X, w, b = cache
    m = X.shape[0]
    dX = dOut.dot(w.T)
    dX = dX.reshape(*X.shape)
    dW = X.reshape(m, -1).T.dot(dOut)
    db = np.sum(dOut, axis=0)

    return dX, dW, db

def relu_forward(X):
    out = np.maximum(0, X)
    cache = X
    return out, cache

def relu_backward(dOut, cache):
    X = cache
    dx = (X >= 0) * dOut
    return dx

def dropout_forward(X, params):
    probability, mode = params['p'], params['mode']

    mask = None
    out = None
    if mode == 'train':
        mask = (np.random.rand(*X.shape) >= probability) / (1 - probability)
        out = X * mask
    elif mode == 'test':
        out = X
    
    cache = (params, mask)
    out = out.astype(X.dtype)
    return out, cache

def dropout_backward(dOut, cache):
    params, mask = cache

    mode = params['mode']
    dX = None

    if mode == 'train':
        dX = dOut * mask
    elif mode == 'test':
        dX = dOut
    
    return dX

def batchnorm_forward(X, gamma, beta, params):
    mode, epsilon, momentum = params['mode'], params.get('eps', 1e-5), params.get('momentum', 0.9)
    
    N, D = X.shape

    running_mean = params.get('running_mean', np.zeros(D, dtype=X.dtype))
    running_variance = params.get('running_var', np.zeros(D, dtype=X.dtype))

    if mode == 'train':
        mu = X.mean(axis=0)
        variance = np.mean((X-mu)**2, axis=0)
        std = np.sqrt(variance + epsilon)
        Xn = (X-mu)/std
        out = gamma * Xn + beta

        cache = (mode, X, gamma, X-mu, std, Xn, out)

        running_mean *= momentum
        running_mean += (1 - momentum) * mu

        running_variance *= momentum
        running_variance += (1 - momentum) * variance
    elif mode == 'test':
        std = np.sqrt(running_variance + epsilon)
        Xn = (X - running_mean) / std
        out = gamma * Xn + beta
        cache = (mode, X, Xn, gamma, beta, std)
    else:
        raise ValueError("Invalid forward batchnorm mode {}".format(mode))
    
    params['running_mean'] = running_mean
    params['running_variance'] = running_variance

    return out, cache

def batchnorm_backward(dOut, cache):
    mode = cache[0]
    if mode == 'train':
        mode, x, gamma, xc, std, xn, out = cache

        N = x.shape[0]
        dbeta = dOut.sum(axis=0)
        dgamma = np.sum(xn * dOut, axis=0)
        dxn = gamma * dOut
        dxc = dxn / std
        dstd = -np.sum((dxn * xc) / (std * std), axis=0)
        dvar = 0.5 * dstd / std
        dxc += (2.0 / N) * xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / N
    elif mode == 'test':
        mode, x, xn, gamma, beta, std = cache
        dbeta = dOut.sum(axis=0)
        dgamma = np.sum(xn * dOut, axis=0)
        dxn = gamma * dOut
        dx = dxn / std
    else:
        raise ValueError(mode)

    return dx, dgamma, dbeta

def spatial_batchnorm_forward(X, gamma, beta, params):
    N, C, W, H = X.shape
    X_flat = X.transpose(0, 2, 3, 1).reshape(-1, C)
    out, cache = batchnorm_forward(X_flat, gamma, beta, params)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X-mu)/sigma, mu, sigma

def spatial_batchnorm_backward(dOut, cache):
    N, C, W, H = dOut.shape
    dOut_flat = dOut.transpose(0, 2, 3, 1).reshape(-1, C)
    dX, dgamma, dbeta = batchnorm_backward(dOut_flat, cache)
    dX = dX.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dX, dgamma, dbeta

def softmax_loss(X, y):
   
    probability = np.exp(X - np.max(X, axis=1, keepdims=True))
    probability /= np.sum(probability, axis=1, keepdims=True)
    N = X.shape[0]
   
    loss = -np.sum(np.log(probability[np.arange(N), y])) / N
    dX = probability.copy()
    dX[np.arange(N), y] -= 1
    dX /= N

    return loss, dX

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    return X_pad

def conv_forward(X, W, b, params):
    N, channels, height, width = X.shape
    n_C, _, f, f = W.shape
    stride = params['stride']
    pad = params['pad']
    n_H = ((height - f + 2*pad) / stride) + 1
    n_W = ((width - f + 2*pad) / stride) + 1

    Z = np.zeros((N, n_C, int(n_H), int(n_W)))
    #Zero padding input to better capture patterns in the edge of an image
    X_pad = zero_pad(X, pad)

    for h in range(int(n_H)):
        for w in range(int(n_W)):
            #build filter range
            vertical_start = h*stride
            vertical_end = vertical_start + f
            horizontal_start = w*stride
            horizontal_end = horizontal_start + f
            X_slice = X_pad[:, :, vertical_start:vertical_end, horizontal_start:horizontal_end]
            for c in range(n_C):
                Z[:, c, h, w] = np.sum(X_slice * W[c, :, :, :], axis=(1,2,3))
    #adding bias term
    Z += (b)[None, :, None, None]
    cache = (X, W, b, params)
    return Z, cache

def conv_backward(dOut, cache):
    X, W, b, params = cache
    N, C, height, width = X.shape
    n_C, _, f, f = W.shape
    stride, pad = params['stride'], params['pad']

    n_H = ((height - f + 2*pad) / stride) + 1
    n_W = ((width - f + 2*pad) / stride) + 1
    X_pad = zero_pad(X, pad)

    dX = np.zeros_like(X)
    dX_pad = np.zeros_like(X_pad)
    dW = np.zeros_like(W)

    db = np.sum(dOut, axis = (0,2,3))

    for h in range(int(n_H)):
        for w in range(int(n_W)):
            #build filter range
            vertical_start = h*stride
            vertical_end = vertical_start + f
            horizontal_start = w*stride
            horizontal_end = horizontal_start + f
            X_slice = X_pad[:, :, vertical_start:vertical_end, horizontal_start:horizontal_end]
            
            #Compute the derivative with respect to W (dW)
            for c in range(n_C):
                dW[c, :, :, :] += np.sum(X_slice * (dOut[:, c, h, w])[:, None, None, None], axis=0)
            
            #Compute the derivative with respect to X_pad (dX_pad)
            for n in range(N):
                dX_pad[n, :, vertical_start:vertical_end, horizontal_start:horizontal_end] += np.sum((W[:, :, :, :]*(dOut[n, :, h, w])[:, None, None, None]), axis=0)
    dX = dX_pad[:, :, pad:-pad, pad:-pad]
    return dX, dW, db

def max_pooling_forward(X, params):
    N, C, H, W = X.shape
    pool_height, pool_width, stride = params['pool_height'], params['pool_width'], params['stride']


    n_H = int(((H - pool_height) / stride) + 1)
    n_W = int(((W - pool_height) / stride) + 1)


    out = np.zeros((N, C, n_H, n_W))

    for h in range(n_H):
        for w in range(n_W):
            #build filter range
            vertical_start = h*stride
            vertical_end = vertical_start + pool_height
            horizontal_start = w*stride
            horizontal_end = horizontal_start + pool_width

            X_masked = X[:, :, vertical_start:vertical_end, horizontal_start:horizontal_end]
            out[:, :, h, w] = np.max(X_masked, axis=(2,3))

    cache = (X, params)
    return out, cache

def max_pooling_backward(dOut, cache):
    X, params = cache
    N, C, H, W = X.shape

    pool_height, pool_width, stride = params['pool_height'], params['pool_width'], params['stride']

    n_H = int(((H - pool_height) / stride) + 1)
    n_W = int(((W - pool_height) / stride) + 1)
    dX = np.zeros_like(X)

    for h in range(n_H):
        for w in range(n_W):
            #build filter range
            vertical_start = h*stride
            vertical_end = vertical_start + pool_height
            horizontal_start = w*stride
            horizontal_end = horizontal_start + pool_width

            X_masked = X[:, :, vertical_start:vertical_end, horizontal_start:horizontal_end]

            max_X_masked = np.max(X_masked, axis=(2,3))
            binary_mask = (X_masked == (max_X_masked)[:, :, None, None])

            dX[:, :, vertical_start:vertical_end, horizontal_start:horizontal_end] += binary_mask * (dOut[:, :, h, w])[:, :, None, None]
    return dX