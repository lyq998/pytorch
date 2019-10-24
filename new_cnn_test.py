import torch
import torch.nn as nn	# 各种层类型的实现
import numpy as np
import torch.nn.functional as F	# 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim	# 实现各种优化算法的包

class Net(nn.ModuleList):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.append(nn.Conv2d(20,10,kernel_size=3))

        for i in self.children():
            print(i)
            print('-----------------------')

model = Net()
model.append(nn.Conv2d(20,10,kernel_size=3))
model.append(nn.Conv2d(20,10,kernel_size=3))
print(model)