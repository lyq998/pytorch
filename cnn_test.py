import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from nn_summary import get_total_params


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(220, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 220, kernel_size=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = self.conv4(x)
        return out

model = Net()

print(get_total_params(model, (220, 50, 50)))

