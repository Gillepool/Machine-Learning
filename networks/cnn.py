import numpy as np
import h5py

from layers import *



class CNN(object):
  def __init__(self, dtype=np.float32, num_classes=10, input_size=28, h5_file=None):
    self.dtype = dtype
    self.conv_params = []
    self.input_size = input_size
    self.num_classes = num_classes
    
  
    self.conv_params.append({'stride': 1, 'pad': 2})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})

    self.filter_sizes = [5, 3, 3, 3]
    self.num_filters = [64, 64, 128, 128]
    hidden_dim = 512

    self.bn_params = []
    
    cur_size = input_size
    prev_dim = 1
    self.params = {}
    for i, (f, next_dim) in enumerate(zip(self.filter_sizes, self.num_filters)):
      fan_in = f * f * prev_dim
      self.params['W%d' % (i + 1)] = np.sqrt(2.0 / fan_in) * np.random.randn(next_dim, prev_dim, f, f)
      self.params['b%d' % (i + 1)] = np.zeros(next_dim)
      self.params['gamma%d' % (i + 1)] = np.ones(next_dim)
      self.params['beta%d' % (i + 1)] = np.zeros(next_dim)
      self.bn_params.append({'mode': 'train'})
      prev_dim = next_dim
      if self.conv_params[i]['stride'] == 2: 
        cur_size /= 2
        print(cur_size)
    
    # Add a fully-connected layers
  
    fan_in = int(cur_size * cur_size * self.num_filters[-1])
    self.params['W%d' % (i + 2)] = np.sqrt(2.0 / fan_in) * np.random.randn(fan_in, hidden_dim)
    self.params['b%d' % (i + 2)] = np.zeros(hidden_dim)
    self.params['gamma%d' % (i + 2)] = np.ones(hidden_dim)
    self.params['beta%d' % (i + 2)] = np.zeros(hidden_dim)
    self.bn_params.append({'mode': 'train'})
    self.params['W%d' % (i + 3)] = np.sqrt(2.0 / hidden_dim) * np.random.randn(hidden_dim, num_classes)
    self.params['b%d' % (i + 3)] = np.zeros(num_classes)
    
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

  
  def forward(self, X, start=None, end=None, mode='test'):
   
    X = X.astype(self.dtype)
    if start is None: start = 0
    if end is None: end = len(self.conv_params) + 1
    layer_caches = []

    prev_a = X
    for i in range(start, end + 1):
      i1 = i + 1
      if 0 <= i < len(self.conv_params):
        # This is a conv layer
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]

        gamma, beta = self.params['gamma%d' % i1], self.params['beta%d' % i1]

        conv_param = self.conv_params[i]
        bn_param = self.bn_params[i]
        bn_param['mode'] = mode
        
        ''' conv  -> spatial batchnorm -> relu '''

        a, conv_cache = conv_forward(prev_a, w, b, conv_param)

        an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
        next_a, relu_cache = relu_forward(an)

        cache = (conv_cache, bn_cache, relu_cache)
      elif i == len(self.conv_params):
        # This is the fully-connected hidden layer
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        gamma, beta = self.params['gamma%d' % i1], self.params['beta%d' % i1]
        bn_param = self.bn_params[i]
        bn_param['mode'] = mode
        ''' affine -> batchnorm -> relu '''

        a, fc_cache = affine_forward(prev_a, w, b)

        a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)

        next_a, relu_cache = relu_forward(a_bn)

        cache = (fc_cache, bn_cache, relu_cache)
      elif i == len(self.conv_params) + 1:
        # This is the last fully-connected layer that produces scores
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        next_a, cache = affine_forward(prev_a, w, b)
      else:
        raise ValueError('Invalid layer index %d' % i)

      layer_caches.append(cache)
      prev_a = next_a

    out = prev_a
    cache = (start, end, layer_caches)
    return out, cache


  def backward(self, dout, cache):
  
    start, end, layer_caches = cache
    dnext_a = dout
    grads = {}
    for i in reversed(range(start, end + 1)):
      i1 = i + 1
      if i == len(self.conv_params) + 1:
        # This is the last fully-connected layer
        dprev_a, dw, db = affine_backward(dnext_a, layer_caches.pop())
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
      elif i == len(self.conv_params):
        # This is the fully-connected hidden layer
        fc_cache, bn_cache, relu_cache = layer_caches.pop()
        ''' relu -> batchnorm -> affine '''
        da_bn = relu_backward(dnext_a, relu_cache)
        da, dgamma, dbeta = batchnorm_backward(da_bn, bn_cache)
        dx, dw, db = affine_backward(da, fc_cache)
        temp = dx, dw, db, dgamma, dbeta

        dprev_a, dw, db, dgamma, dbeta = temp
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
        grads['gamma%d' % i1] = dgamma
        grads['beta%d' % i1] = dbeta
      elif 0 <= i < len(self.conv_params):
        # This is a conv layer
        conv_cache, bn_cache, relu_cache = layer_caches.pop()
        dan = relu_backward(dnext_a, relu_cache)
        da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
        dx, dw, db = conv_backward(da, conv_cache)

        temp = dx, dw, db, dgamma, dbeta

        dprev_a, dw, db, dgamma, dbeta = temp
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
        grads['gamma%d' % i1] = dgamma
        grads['beta%d' % i1] = dbeta
      else:
        raise ValueError('Invalid layer index %d' % i)
      dnext_a = dprev_a

    dX = dnext_a
    return dX, grads


  def loss(self, X, y=None):
  
    mode = 'test' if y is None else 'train'
    scores, cache = self.forward(X, mode=mode)
    if mode == 'test':
      return scores
    loss, dscores = softmax_loss(scores, y)
    dX, grads = self.backward(dscores, cache)
    return loss, grads