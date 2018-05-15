# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
      

        #print("gamma : ")
        #print(self.gamma)
        #print("\nbeta :")
        #print(self.beta)
        #print("\n")

        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class BatchNormalization_linearized:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, k, l, momentum=0.9, running_mean=None, running_var=None):
        self.k = k
        self.l = l
        #self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        #self.running_mean = running_mean
        #self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        #self.xc = None
        #self.std = None
        self.dk = None
        self.dl = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        #if self.running_mean is None:
        #    N, D = x.shape
        #    self.running_mean = np.zeros(D)
        #    self.running_var = np.zeros(D)
                        
        if train_flg:
            #mu = x.mean(axis=0)
            #xc = x - mu
            #var = np.mean(xc**2, axis=0)
            #std = np.sqrt(var + 10e-7)
            #xn = xc / std
            self.x = x
            self.batch_size = x.shape[0]
            #self.xc = xc
            #self.xn = xn
            #self.std = std
            #self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            #self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        #else:
            #xc = x - self.running_mean
            #xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.k * x + self.l 
        
        print("k : ")
        print(self.k)
        print("\nl :")
        print(self.l)
        print("\n")
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):

        N = dout.shape[0]
        dl = dout.sum(axis=0)
        dk = np.sum(self.x * dout, axis=0)
        #dxn = self.gamma * dout
        #dxc = dxn / self.std
        #dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        #dvar = 0.5 * dstd / self.std
        #dxc += (2.0 / self.batch_size) * self.xc * dvar
        #dmu = np.sum(dxc, axis=0)
        #dx = dxc - dmu / self.batch_size
        dx = self.k * dout
        self.dk = dk/N
        self.dl = dl/N
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        # np.shape(dout) = (N, FN, out_h, out_w)
        # transpose(0,2,3,1) = (N, out_h, out_w, FN)
        # reshape(-1, FN) = (N*out_h*out_w, FN)
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        # np.shape(col.T) = (C*FH*FW,N*out_h*out_w)
        # dW = (C*FH*FW,FN)
        self.dW = np.dot(self.col.T, dout)
        # transpose(1, 0) = (FN, C*FH*FW)
        # reshape(FN, C, FH, FW)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # np.shape(col_w.T) = (FN, C*FH*FW)
        # dcol = (N*out_h*out_w, C*FH*FW)
        dcol = np.dot(dout, self.col_W.T)
        # x.shape = N, C, H, W
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        # (N C Hout Wout) into (N Hout Wout C)
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        
        dcol = dmax.reshape(dout.shape[0] * dout.shape[1] * dout.shape[2], -1)

        ############
        # test code#
        ############
        # dmaxbuf=dmax;
        # original
        # dmax = dmax.reshape(dout.shape + (pool_size,)) 
        # dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        
        # edited
        #dcolbuf = dmaxbuf.reshape(dout.shape[0] * dout.shape[1] * dout.shape[2], -1) 

        # z = (dcol==dcolbuf)
        # if (np.sum(z)==dcol.size and np.sum(z)==dcolbuf.size):
        #    print('it is exactly same')   
        #    print('np sum(z): '+str(np.sum(z)))
        #    print('dcol.size : '+str(dcol.size))
        #    print('dcolbuf.size : '+str(dcolbuf.size))
        # else : print(nocool)


        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
