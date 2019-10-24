import numpy as np
from population import *
import copy
from evaluate import Evaluate


class Evolve_CNN:
    def __init__(self, m_prob, m_eta, x_prob, x_eta, population_size, train_data, train_label, validate_data,
                 validate_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, eta):
        self.m_prob = m_prob
        self.m_eta = m_eta
        self.x_prob = x_prob
        self.x_eta = x_eta
        self.population_size = population_size
        self.train_data = train_data
        self.train_label = train_label
        self.validate_data = validate_data
        self.validate_label = validate_label
        self.epochs = epochs
        self.eta = eta
        self.number_of_channel = number_of_channel
        self.batch_size = batch_size
        self.train_data_length = train_data_length
        self.validate_data_length = validate_data_length

    def initialize_popualtion(self):
        print("initializing population with number {}...".format(self.population_size))
        self.pops = Population(self.population_size)
        # all the initialized population should be saved
        save_populations(gen_no=-1, pops=self.pops)

    def evaluate_fitness(self, gen_no):
        print("evaluate fintesss")
        evaluate = Evaluate(self.pops, self.train_data, self.train_label, self.validate_data, self.validate_label,
                            self.number_of_channel, self.epochs, self.batch_size, self.train_data_length,
                            self.validate_data_length)
        evaluate.parse_population(gen_no)
        #         # all theinitialized population should be saved
        save_populations(gen_no=gen_no, pops=self.pops)
        print(self.pops)

    def recombinate(self, gen_no):
        print("mutation and crossover...")
        offspring_list = []
        for _ in range(int(self.pops.get_pop_size() / 2)):
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            # crossover
            offset1, offset2 = self.crossover(p1, p2)
            # mutation
            offset1.mutation()
            offset2.mutation()
            offspring_list.append(offset1)
            offspring_list.append(offset2)
        offspring_pops = Population(0)
        offspring_pops.set_populations(offspring_list)
        save_offspring(gen_no, offspring_pops)
        # evaluate these individuals
        evaluate = Evaluate(self.pops, self.train_data, self.train_label, self.validate_data, self.validate_label,
                            self.number_of_channel, self.epochs, self.batch_size, self.train_data_length,
                            self.validate_data_length)
        evaluate.parse_population(gen_no)
        #         #save
        self.pops.pops.extend(offspring_pops.pops)
        save_populations(gen_no=gen_no, pops=self.pops)

    def crossover(self, p1, p2):
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        p1.clear_state_info()
        p2.clear_state_info()

        p1_conv_layer_list = []
        p2_conv_layer_list = []

        for i in range(p1.get_layer_size()):
            p1_conv_layer_list.append(p1.get_layer_at(i))
        for i in range(p2.get_layer_size()):
            p2_conv_layer_list.append(p2.get_layer_at(i))

        l = min(len(p1_conv_layer_list), len(p2_conv_layer_list))
        for i in range(l):
            unit_p1 = p1_conv_layer_list[i]
            unit_p2 = p2_conv_layer_list[i]
            if flip(self.x_prob):
                # filter size : exchange each's filter size
                w1 = unit_p1.filter_width
                w2 = unit_p2.filter_width
                unit_p1.filter_width = w2
                unit_p1.filter_height = w2
                unit_p2.filter_width = w1
                unit_p2.filter_height = w1
                # feature map size
                this_range = p1.featur_map_size_range
                s1 = unit_p1.feature_map_size
                s2 = unit_p2.feature_map_size
                n_s1, n_s2 = self.sbx(s1, s2, this_range[0], this_range[-1], self.x_eta)
                unit_p1.feature_map_size = int(n_s1)
                unit_p2.feature_map_size = int(n_s2)
                # weight_matrix_mean
                this_range = p1.mean_range
                m1 = unit_p1.weight_matrix_mean
                m2 = unit_p2.weight_matrix_mean
                n_m1, n_m2 = self.sbx(m1, m2, this_range[0], this_range[-1], self.x_eta)
                unit_p1.weight_matrix_mean = n_m1
                unit_p2.weight_matrix_mean = n_m2
                # weight_matrix_std
                this_range = p1.std_range
                std1 = unit_p1.weight_matrix_std
                std2 = unit_p2.weight_matrix_std
                n_std1, n_std2 = self.sbx(std1, std2, this_range[0], this_range[-1], self.x_eta)
                unit_p1.weight_matrix_std = n_std1
                unit_p2.weight_matrix_std = n_std2
            p1_conv_layer_list[i] = unit_p1
            p2_conv_layer_list[i] = unit_p2

        p1.indi = p1_conv_layer_list
        p2.indi = p2_conv_layer_list

        return p1, p2

    def environmental_selection(self, gen_no):
        assert (self.pops.get_pop_size() == 2 * self.population_size)
        elitsam = 0.2
        e_count = int(np.floor(self.population_size * elitsam / 2) * 2)
        indi_list = self.pops.pops
        indi_list.sort(key=lambda x: x.mean, reverse=True)
        elistm_list = indi_list[0:e_count]

        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)
        np.random.shuffle(left_list)

        for _ in range(self.population_size - e_count):
            i1 = randint(0, len(left_list))
            i2 = randint(0, len(left_list))
            winner = self.selection(left_list[i1], left_list[i2])
            elistm_list.append(winner)

        self.pops.set_populations(elistm_list)
        save_populations(gen_no=gen_no, pops=self.pops)
        np.random.shuffle(self.pops.pops)

    def sbx(self, p1, p2, xl, xu, eta):
        '''
        :param self:
        :param p1: 父亲1
        :param p2: 父亲2
        :param xl: 最小值
        :param xu: 最大值
        :param eta: sbx的参数（建议取1）
        :return: 两个交叉后的子代
        '''
        # par1为更大的那个父亲
        if p1 > p2:
            par1 = p1
            par2 = p2
        else:
            par1 = p2
            par2 = p1
        yl = xl
        yu = xu
        rand = np.random.random()
        if rand <= 0.5:
            betaq = (2 * rand) ** (1 / (eta + 1))
        else:
            betaq = (1 / (2 - 2 * rand)) ** (1 / (eta + 1))
        child1 = 0.5 * ((par1 + par2) - betaq * (par1 - par2))
        child2 = 0.5 * ((par1 + par2) + betaq * (par1 - par2))
        if child1 < yl:
            child1 = yl
        if child1 > yu:
            child1 = yu
        if child2 < yl:
            child2 = yl
        if child2 > yu:
            child2 = yu
        return child1, child2

    def tournament_selection(self):
        ind1_id = randint(0, self.pops.get_pop_size())
        ind2_id = randint(0, self.pops.get_pop_size())
        ind1 = self.pops.get_individual_at(ind1_id)
        ind2 = self.pops.get_individual_at(ind2_id)
        winner = self.selection(ind1, ind2)
        return winner

    def selection(self, ind1, ind2):
        # Slack Binary Tournament Selection
        mean_threshold = 0.05
        complexity_threhold = 100000
        # 一个四层的cnn（fliter_size均为3的话，param个数为220万）
        if ind1.mean > ind2.mean:
            if ind1.mean - ind2.mean > mean_threshold:
                return ind1
            else:
                if ind2.complxity < (ind1.complxity - complexity_threhold):
                    return ind2
                else:
                    return ind1
        else:
            if ind2.mean - ind1.mean > mean_threshold:
                return ind2
            else:
                if ind1.complxity < (ind2.complxity - complexity_threhold):
                    return ind1
                else:
                    return ind2


if __name__ == '__main__':
    cnn = Evolve_CNN(0.1, 10, 0.2, 1, 10, 1, 11, 1, 1, 1, 1, 1, 1, 1, 1)
    cnn.initialize_popualtion()
    print("a")
