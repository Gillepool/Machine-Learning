import numpy as np
from layers import *

class FullyConnectedNet(object):
  def __init__(self, hidden_dims, input_dim=1*28*28, num_classes=10,
               dropout=0.0, use_batchnorm=True, reg=0.01,
               weight_scale=1e-4, dtype=np.float32, seed=None):
  
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
   
    layer_input_dim = input_dim
    for i, hd in enumerate(hidden_dims):
        self.params['W%d'%(i+1)] = weight_scale * np.random.randn(layer_input_dim, hd)
        self.params['b%d'%(i+1)] = weight_scale * np.zeros(hd)
        if self.use_batchnorm:
            self.params['gamma%d'%(i+1)] = np.ones(hd)
            self.params['beta%d'%(i+1)] = np.zeros(hd)
        layer_input_dim = hd
    self.params['W%d'%(self.num_layers)] = weight_scale * np.random.randn(layer_input_dim, num_classes)
    self.params['b%d'%(self.num_layers)] = weight_scale * np.zeros(num_classes)

   
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
  
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode 
        
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    scores = None

    layer_input = X
    ar_cache = {}
    dp_cache = {}
    
    for lay in range(self.num_layers-1):
        if self.use_batchnorm:
            a, fc_cache = affine_forward(layer_input,  self.params['W%d'%(lay+1)], self.params['b%d'%(lay+1)])
            a_bn, bn_cache = batchnorm_forward(a, self.params['gamma%d'%(lay+1)], self.params['beta%d'%(lay+1)], self.bn_params[lay])
            layer_input, relu_cache = relu_forward(a_bn)
            ar_cache[lay] = (fc_cache, bn_cache, relu_cache)
        else:
            a, fc_cache = affine_forward(layer_input, self.params['W%d'%(lay+1)], self.params['b%d'%(lay+1)])
            layer_input, relu_cache = relu_forward(a)
            ar_cache[lay] = (fc_cache, relu_cache)
            
        if self.use_dropout:
            layer_input,  dp_cache[lay] = dropout_forward(layer_input, self.dropout_param)
            
    ar_out, ar_cache[self.num_layers] = affine_forward(layer_input, self.params['W%d'%(self.num_layers)], self.params['b%d'%(self.num_layers)])
    scores = ar_out

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    loss, dscores = softmax_loss(scores, y)
    dhout = dscores
    loss = loss + 0.5 * self.reg * np.sum(self.params['W%d'%(self.num_layers)] * self.params['W%d'%(self.num_layers)])
    dx , dw , db = affine_backward(dhout , ar_cache[self.num_layers])
    grads['W%d'%(self.num_layers)] = dw + self.reg * self.params['W%d'%(self.num_layers)]
    grads['b%d'%(self.num_layers)] = db
    dhout = dx
    for idx in range(self.num_layers-1):
        lay = self.num_layers - 1 - idx - 1
        loss = loss + 0.5 * self.reg * np.sum(self.params['W%d'%(lay+1)] * self.params['W%d'%(lay+1)])
        if self.use_dropout:
            dhout = dropout_backward(dhout ,dp_cache[lay])
        if self.use_batchnorm:
            fc_cache, bn_cache, relu_cache = ar_cache[lay]
            da_bn = relu_backward(dhout, relu_cache)
            da, dgamma, dbeta = batchnorm_backward(da_bn, bn_cache)
            dx, dw, db = affine_backward(da, fc_cache)
        else:
            fc_cache, relu_cache = ar_cache[lay]
            da = relu_backward(dhout, relu_cache)
            dx, dw, db = affine_backward(da, fc_cache)

        grads['W%d'%(lay+1)] = dw + self.reg * self.params['W%d'%(lay+1)]
        grads['b%d'%(lay+1)] = db
        if self.use_batchnorm:
            grads['gamma%d'%(lay+1)] = dgamma
            grads['beta%d'%(lay+1)] = dbeta
        dhout = dx
   
    return loss, grads