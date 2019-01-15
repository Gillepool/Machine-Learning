from keras.datasets import fashion_mnist, mnist
import numpy as np
import os


def load_mnist_data(num_training=59000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    print ('Loading data....')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(X_train.shape)

    X_train = X_train.reshape(60000, 1, 28, 28).transpose(0, 2, 3, 1).astype('float')
    y_train = np.array(y_train)

    y_test = np.array(y_test)
    X_test = X_test.reshape(10000, 1, 28, 28).transpose(0, 2, 3, 1).astype('float')

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
      mean_image = np.mean(X_train, axis=0)
      X_train -= mean_image
      X_val -= mean_image
      X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
