import numpy as np

def sgd(W, dW, config={}):
    "Stochastic gradient descent"
    config.setdefault('learning_rate', 1e-2)

    W -= config['learning_rate'] * dW
    return W, config

def sgd_momentum(W, dW, config={}):
    "Stochastic gradient descent with momentum"
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    velocity = config.get('velocity', np.zeros_like(W))

    velocity = config['momentum'] * velocity - config['learning_rate'] * dW

    next_W = W + velocity
    config['velocity'] = velocity
    return next_W, config

def rmsprop(X, dX, config={}):
    "RMSProp, using the moving average of the squared gradient values"

    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(X))

    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * (dX**2)
    next_X = X - config['learning_rate'] * dX / (np.sqrt(config['cache']) + config['epsilon'])

    return next_X, config


def adam(X, dX, config={}):
    "Adams optimziers, using the moving average of both the gradient and its square while adding a bias term"
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(X))
    config.setdefault('v', np.zeros_like(X))
    config.setdefault('t', 0)

    next_x = None
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    t, m, v = config['t'], config['m'], config['v']


    m = beta1 * m + (1 - beta1) * dX
    v = beta2 * v + (1 - beta2) * (dX * dX)
    t += 1
    alpha = config['learning_rate'] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    X -= alpha * (m / (np.sqrt(v) + eps))
    config['t'] = t
    config['m'] = m
    config['v'] = v
    next_x = X


    return next_x, config

