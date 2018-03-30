# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        print('in net class, self.W:\n'+str(self.W))
        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])


net = simpleNet()

print ('net.W : \n'+str(net.W))
#
#f = lambda w: net.loss(x, t)

def f(W):
	print('in f, W :\n'+str(W))
	print('in f, x :\n'+str(x))
	print('in f, t :\n'+str(t))
	return net.loss(x,t)

dW = numerical_gradient(f, net.W)

print(dW)

print ('net.W : \n'+str(net.W))