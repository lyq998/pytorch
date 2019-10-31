import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import get_data
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(220, 256, 3)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 220, 3)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.conv4(x)
        # return x
        for layer in self.children():
            x = layer(x)
        return x


if __name__ == '__main__':
    loss_dict = []
    net = Net()
    net.cuda()
    print(net)
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(),lr=0.004)

    train_loader = get_data.get_train_loader(16)
    num_epochs = train_loader.__len__()

    for i, data in enumerate(train_loader, 0):
        # Convert numpy arrays to torch tensors  5.1 准备tensor的训练数据和标签
        inputs, labels = data
        labels = get_data.get_size_labels(4, labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        # labels = get_data.get_size_labels(indi.get_layer_size(),labels)

        # Forward pass  5.2 前向传播计算网络结构的输出结果
        optimizer.zero_grad()
        outputs = net(inputs)
        # 5.3 计算损失函数
        loss = criterion(outputs, labels)
        loss = loss.cuda()

        # Backward and optimize 5.4 反向传播更新参数
        loss.backward()
        optimizer.step()
        loss_dict.append(loss.item())

        # 可选 5.5 打印训练信息和保存loss
        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(i + 1, num_epochs, loss.item()))

    for layer in net.children():
        if isinstance(layer, nn.Conv2d):
            print('layer_mean:', layer.weight.mean())
            print('layer_std:', layer.weight.std())
        if isinstance(layer,nn.BatchNorm2d):
            print('bn_mean:',layer.weight.mean())
            print('bn_std:', layer.weight.std())
    plt.plot(loss_dict[500:])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
