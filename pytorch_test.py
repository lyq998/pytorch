import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.padding1 = nn.ReflectionPad2d(1)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x= self.padding1(x)
        return x
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2_drop(self.conv2(x)))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)


m = nn.ReflectionPad2d(2)
input = torch.arange(9,dtype=torch.float).reshape(1, 1, 3, 3)
net = Net()
output=net(input)
print(output)