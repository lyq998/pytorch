import numpy as np


class ConvLayer:
    def __init__(self, filter_size=[3, 3], feature_map_size=220, weight_matrix=[0.0, 0.5]):
        self.filter_width = filter_size[0]
        self.filter_height = filter_size[1]
        self.feature_map_size = feature_map_size
        self.weight_matrix_mean = weight_matrix[0]
        self.weight_matrix_std = weight_matrix[1]
        self.type = 1
        # 设conv.type=1

    def __str__(self):
        return "Conv Layer: filter:[{0},{1}], feature map number:{2}, weight:[{3},{4}]".format(self.filter_width,
                                                                                               self.filter_height,
                                                                                               self.feature_map_size,
                                                                                               self.weight_matrix_mean,
                                                                                               self.weight_matrix_std)


class BatchNormLayer:
    def __init__(self, weight_matrix=[1.0, 0.5]):
        self.weight_matrix_mean = weight_matrix[0]
        self.weight_matrix_std = weight_matrix[1]
        self.type = 2
        # 设batchnorm.type=2

    def __str__(self):
        return "BatchNorm Layer: weight:[{},{}]".format(self.weight_matrix_mean,
                                                        self.weight_matrix_std)
