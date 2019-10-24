import numpy as np
from layers import *
from utils import *


class Individual:
    def __init__(self, x_prob=0.9, x_eta=1, m_prob=0.2, m_eta=1):
        # x_prob是sbx概率，x_eta是sbx参数，同理m代表pm
        self.indi = []
        self.x_prob = x_prob
        self.x_eta = x_eta
        self.m_prob = m_prob
        self.m_eta = m_eta
        self.mean = 0
        self.std = 0
        self.complexity = 0  # 复杂度，用number of params来衡量

        #####################
        self.feature_map_size_range = [128, 512]
        self.filter_size_set = [1, 3]  # len(filter_size_set为2，表示只有两个可能：1或3)
        self.mean_range = [-1, 1]
        self.std_range = [0, 1]

    def initialize(self):
        self.indi = self.init_one_individual()

    def init_one_individual(self):
        init_num_conv = np.random.randint(4, 8)
        _list = []
        for _ in range(init_num_conv-1):
            _list.append(self.add_a_random_conv_layer())
        _list.append(self.add_a_last_conv_layer())
        return _list

    def get_layer_at(self, i):
        return self.indi[i]

    def get_layer_size(self):
        return len(self.indi)

    def init_mean(self):
        return np.random.random() * (self.mean_range[1] - self.mean_range[0]) + self.mean_range[0]

    def init_std(self):
        return np.random.random() * (self.std_range[1] - self.std_range[0]) + self.std_range[0]

    def init_feature_map_size(self):
        return np.random.randint(self.feature_map_size_range[0], self.feature_map_size_range[1])

    def init_kernel_size(self):
        kernel_size_num = len(self.filter_size_set)
        n = np.random.randint(kernel_size_num)
        return self.filter_size_set[n]

    def add_a_random_conv_layer(self):
        s1 = self.init_kernel_size()
        filter_size = s1, s1
        feature_map_size = self.init_feature_map_size()
        mean = self.init_mean()
        std = self.init_std()
        conv_layer = ConvLayer(filter_size=filter_size, feature_map_size=feature_map_size, weight_matrix=[mean, std])
        return conv_layer

    def add_a_last_conv_layer(self):
        s1 = self.init_kernel_size()
        filter_size = s1, s1
        feature_map_size = 220
        mean = self.init_mean()
        std = self.init_std()
        conv_layer = ConvLayer(filter_size=filter_size, feature_map_size=feature_map_size, weight_matrix=[mean, std])
        return conv_layer

    def mutation(self):
        if flip(self.m_prob):
            # for the units
            unit_list = []
            for i in range(self.get_layer_size() - 1):
                cur_unit = self.get_layer_at(i)
                if flip(0.5):
                    # mutation
                    p_op = self.mutation_ope(rand())
                    min_length = 4
                    max_length = 8
                    current_length = (len(unit_list) + self.get_layer_size() - i - 1)  #current_length是现在unit_list长度加剩下去掉最后一层的长度，所以下面是小于而不是小于等于
                    if p_op == 0:  # add a new

                        if current_length < max_length:  # when length exceeds this length, only mutation no add new unit
                            unit_list.append(self.generate_a_new_layer())
                            unit_list.append(cur_unit)
                        else:
                            updated_unit = self.mutation_a_unit(cur_unit, self.m_eta)
                            unit_list.append(updated_unit)
                    if p_op == 1:  # modify the element
                        updated_unit = self.mutation_a_unit(cur_unit, self.m_eta)
                        unit_list.append(updated_unit)
                    if p_op == 2:  # delete the element
                        if current_length < min_length:
                            # when length not exceeds this length, only mutation no add new unit
                            updated_unit = self.mutation_a_unit(cur_unit, self.m_eta)
                            unit_list.append(updated_unit)
                        # else: delete -> don't append the unit into unit_list -> do nothing

                else:
                    unit_list.append(cur_unit)
            # avoid all units have been removed, add a full layer
            if len(unit_list) == 0:
                unit_list.append(self.add_a_random_conv_layer())
            # 最后一层不去动他，这样就保证输出结果格式的正确性。前面的for i 是从1开始
            unit_list.append(self.get_layer_at(-1))
            # judge the first unit and the second unit
            if unit_list[0].type != 1:
                unit_list.insert(0, self.add_a_random_conv_layer())
            self.indi = unit_list

    def mutation_a_unit(self, unit, eta):
        if unit.type == 1:
            # mutate a conv layer
            return self.mutate_conv_unit(unit, eta)

    def mutate_conv_unit(self, unit, eta):
        # filter size, feature map number, mean std
        # fs = unit.filter_width    孙老师要用这一步进行下面的Mutation，我是固定的大小1,3 -> use np.random.choice
        fmn = unit.feature_map_size
        mean = unit.weight_matrix_mean
        std = unit.weight_matrix_std

        new_fs = np.random.choice(self.filter_size_set)
        new_fmn = int(self.pm(self.feature_map_size_range[0], self.feature_map_size_range[1], fmn, eta))
        new_mean = self.pm(self.mean_range[0], self.mean_range[1], mean, eta)
        new_std = self.pm(self.std_range[0], self.std_range[1], std, eta)
        conv_layer = ConvLayer(filter_size=[new_fs, new_fs], feature_map_size=new_fmn,
                               weight_matrix=[new_mean, new_std])
        return conv_layer

    def mutation_ope(self, r):
        # 0 add, 1 modify  2delete
        if r < 0.33:
            return 1
        elif r > 0.66:
            return 2
        else:
            return 0

    def generate_a_new_layer(self):
        return self.add_a_random_conv_layer()

    def pm(self, xl, xu, x, eta):
        '''
        :param xl: 最小值
        :param xu: 最大值
        :param x: 需要多项式变异的实数
        :param eta: pm的参数（更愿意取10）
        :return: pm变异后的实数
        '''
        y = x
        yl = xl
        yu = xu
        y_eta = eta
        delta1 = (y - yl) / (yu - yl)
        delta2 = (yu - y) / (yu - yl)
        rand = np.random.random()
        if rand <= 0.5:
            val = 2 * rand + (1 - 2 * rand) * (1 - delta1) ** (y_eta + 1)
            deltaq = val ** (1 / (y_eta + 1)) - 1
        else:
            val = 2 * (1 - rand) + (2 * rand - 1) * (1 - delta2) ** (y_eta + 1)
            deltaq = 1 - val ** (1 / (y_eta + 1))
        y = y + deltaq * (yu - yl)
        if y < yl:
            y = yl
        if y > yu:
            y = yu
        return y


if __name__ == "__main__":
    indi = Individual()
    # for _ in range(10):
    # print(i.pm(1, 3, 1, 10))
    # print(np.random.choice(i.filter_size_range))
    # print(len(i.filter_size_range))
    # print(i.init_kernel_size())

    # indi.initialize()
    # print(indi.get_layer_size())
    # for i in range(indi.get_layer_size()):
    #     cur_unit = indi.get_layer_at(i)
    #     print(cur_unit)
    #     print(cur_unit.type)
    #     print('------------------')
    #
    # print('------------------------')
    #
    # indi.mutation()
    # for i in range(indi.get_layer_size()):
    #     cur_unit = indi.get_layer_at(i)
    #     print(cur_unit)
    #     print('------------------')
