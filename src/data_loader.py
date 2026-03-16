# import os
# import sys
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')

from keras.datasets import mnist, fashion_mnist
import numpy as np

dataset = fashion_mnist

(X_train, y_train), (X_test, y_test) = dataset.load_data()
X_train = X_train.reshape(60000, 784).astype(np.float64) / 255.0
X_test = X_test.reshape(10000, 784).astype(np.float64) / 255.0

y_train = np.eye(10)[y_train]  # (60000, 10)
y_test = np.eye(10)[y_test]  # (10000, 10)
