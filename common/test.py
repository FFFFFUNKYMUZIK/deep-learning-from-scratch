from optimizer import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as mplp 


#optimizer=Momentum()
optimizer=RMSprop(0.1)

x0=np.array([-7.0, 2.0]);

def gradient(x):
	y={}
	y['x0']=1/10*x['x0']
	y['x1']=2*x['x1']
	return y

x_list=[x0[0]];
y_list=[x0[1]];

x={}
x['x0']=x0[0]
x['x1']=x0[1]

xp=x0

for i in range(1000):

	grad=gradient(x)
	optimizer.update(x, grad)
	if i == 1 or i == 99:
		print('x : ')
		print(x)
	
	x_list.append(x['x0'])
	y_list.append(x['x1'])
	

def plotcontour(xlim,ylim):
	x=list(range(-xlim*1000,xlim*1000))*0.001
	y=list(range(-ylim*1000,ylim*1000))*0.001
	mplp.plot(x,y)
	mplp.show()

plt.plot(x_list, y_list, label='optimizer', linestyle='--', marker = 'o')
plt.show()
