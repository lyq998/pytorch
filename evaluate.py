import numpy as np
from cnn import CNN
import torch.nn as nn
import torch
import get_data
from nn_summary import get_total_params
from torch.autograd import Variable
import os
import pickle
import utils
import matplotlib.pyplot as plt


class Evaluate:

    def __init__(self, pops, batch_size):
        self.pops = pops
        self.batch_size = batch_size

    def parse_population(self, gen_no, evaluated_num):
        save_dir = os.getcwd() + '/save_data/gen_{:03d}'.format(gen_no)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        for i in range(evaluated_num, self.pops.get_pop_size()):
            indi = self.pops.get_individual_at(i)
            rs_mean_loss, rs_std, num_connections = self.parse_individual(indi)
            indi.mean_loss = rs_mean_loss
            indi.std = rs_std
            indi.complexity = num_connections
            list_save_path = os.getcwd() + '/save_data/gen_{:03d}/pop.txt'.format(gen_no)
            utils.save_append_individual(str(indi), list_save_path)
            utils.save_populations(gen_no, self.pops)

        utils.save_generated_population(gen_no, self.pops)

    def parse_individual(self, indi):
        torch_device = torch.device('cuda')
        cnn = CNN(indi)
        cnn.cuda()
        print(cnn)
        complexity = get_total_params(cnn.cuda(), (220, 30, 30))

        train_loader = get_data.get_train_loader(self.batch_size)

        # Loss and optimizer 3.定义损失函数， 使用的是最小平方误差函数
        criterion = nn.MSELoss()
        criterion = criterion.to(torch_device)

        # 4.定义迭代优化算法， 使用的是Adam，SGD不行
        learning_rate = 0.004
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
        loss_dict = []
        num_epochs = train_loader.__len__()
        # Train the model 5. 迭代训练
        cnn.train()
        for i, data in enumerate(train_loader, 0):
            # Convert numpy arrays to torch tensors  5.1 准备tensor的训练数据和标签
            inputs, labels = data
            labels = get_data.get_size_labels(1, labels)
            inputs = inputs.cuda()
            labels = labels.cuda()
            # labels = get_data.get_size_labels(indi.get_layer_size(),labels)

            # Forward pass  5.2 前向传播计算网络结构的输出结果
            optimizer.zero_grad()
            outputs = cnn(inputs)
            # 5.3 计算损失函数
            loss = criterion(outputs, labels)
            loss = loss.cuda()

            # Backward and optimize 5.4 反向传播更新参数
            loss.backward()
            optimizer.step()

            # 可选 5.5 打印训练信息和保存loss
            loss_dict.append(loss.item())
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(i + 1, num_epochs, loss.item()))

        # evaluate
        cnn.eval()
        eval_loss_dict = []
        valid_loader = get_data.get_validate_loader(self.batch_size)
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            labels = get_data.get_size_labels(1, labels)
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss = loss.cuda()
            eval_loss_dict.append(loss.item())

        mean_test_loss = np.mean(eval_loss_dict)
        std_test_loss = np.std(eval_loss_dict)
        print("valid mean:{},std:{}".format(mean_test_loss, std_test_loss))
        return mean_test_loss, std_test_loss, complexity

        # return mean_test_accu, np.std(test_accuracy_list), complexity, history_best_score


if __name__ == '__main__':
    import evolve

    cnn = evolve.Evolve_CNN(0.1, 10, 0.2, 1, 10, 8)
    cnn.initialize_popualtion()

    evaluate = Evaluate(cnn.pops, cnn.batch_size)
    evaluate.parse_population(0)
