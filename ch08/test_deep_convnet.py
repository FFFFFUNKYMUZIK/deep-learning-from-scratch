# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 処理に時間のかかる場合はデータを削減 
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = DeepConvNet( input_dim=(1,28,28), 
                       conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 		conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 		conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 		conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 		conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 		conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 		hidden_size=50, output_size=10)
                        
         
#Train stage ( time consuming )                       
#trainer = Trainer(network, x_train, t_train, x_test, t_test,
#                  epochs=max_epochs, mini_batch_size=100,
#                  optimizer='Adam', optimizer_param={'lr': 0.001},
#                  evaluate_sample_num_per_epoch=1000)
#trainer.train()

# Save Network parameters
# network.save_params("params.pkl")
# print("Saved Network Parameters!")




# Load Network parameters
network.load_params("deep_convnet_params.pkl")
print( 'deep_conv_net_params.pkl is loaded!')

# accuracy calculation
train_acc_list = []
test_acc_list = []
print( 'accuracy calculation is in progress...')
train_acc = network.accuracy(x_train, t_train)
test_acc = network.accuracy(x_test, t_test)

train_acc_list.append(train_acc)
test_acc_list.append(test_acc)
print( 'accuracy calculation is complete.')

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(1)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
