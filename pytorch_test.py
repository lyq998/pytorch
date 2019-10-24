import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from nn_summary import get_total_params


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dnn = nn.ModuleList()
        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        for layer in self.dnn:
            x = layer(x)
        return x
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2_drop(self.conv2(x)))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)


model = Net()
model.dnn.extend(nn.Conv2d(3, 3, 1, 1) for i in range(3))
# model.dnn.append(nn.Conv2d(10, 10, 1, 1))
print(model)
