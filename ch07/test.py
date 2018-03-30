import sys, os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np


# 10 data, 3 chns, 320x240
x1=np.random.randn(10, 3, 320, 240)
# 10 data, 6 chns, 80x60
x2=np.random.randn(10, 6, 80, 60)


#(80080=floor((320-10)/3+1)*floor((240-10)/3+1), 300)
x1_flatten = im2col(x1, filter_h = 10, filter_w = 10, stride = 3, pad = 0)

#(1920=10*16*12, 150)
x2_flatten = im2col(x2, filter_h = 5, filter_w = 5, stride = 5, pad = 0)

print(np.shape(x1_flatten))
print(np.shape(x2_flatten))
