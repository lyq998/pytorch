import torch.nn as nn
import torch


class CNN(nn.ModuleList):
    def __init__(self,indi):
        super(CNN, self).__init__()
        num_of_units = indi.get_layer_size()
        in_channels = 220  # 初始输入图像通道为220
        out_channels = 220

        for i in range(num_of_units):
            # self.append(nn.Conv2d(in_channels,in_channels,3))
            # self.append(nn.ReLU())
            current_unit = indi.get_layer_at(i)
            if current_unit.type == 1:
                filter_size = [current_unit.filter_width, current_unit.filter_height]
                mean = current_unit.weight_matrix_mean
                std = current_unit.weight_matrix_std
                out_channels = current_unit.feature_map_size
                conv = nn.Conv2d(in_channels, out_channels, filter_size)
                nn.init.normal_(conv.weight, mean, std)
                conv.weight.requires_grad = True
                self.append(conv)

                in_channels = current_unit.feature_map_size
            elif current_unit.type == 2:
                mean = current_unit.weight_matrix_mean
                std = current_unit.weight_matrix_std
                batchnorm = nn.BatchNorm2d(out_channels)
                nn.init.normal_(batchnorm.weight, mean, std)
                batchnorm.weight.requires_grad = True
                self.append(batchnorm)
                self.append(nn.ReLU())
                # 在每个卷积层后面加上ReLU激活函数
            else:
                raise NameError('No unit with type value {}'.format(current_unit.type))

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x