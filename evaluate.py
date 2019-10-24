import numpy as np
from cnn import CNN
import torch.nn as nn
import torch
import get_data
from nn_summary import get_total_params
from torch.autograd import Variable


class Evaluate:

    def __init__(self, pops, train_data, train_label, validate_data, validate_label, number_of_channel, epochs,
                 batch_size, train_data_length, validate_data_length):
        self.pops = pops
        self.train_data = train_data  # train or test data.
        self.train_label = train_label
        self.validate_data = validate_data
        self.validate_label = validate_label
        self.number_of_channel = number_of_channel
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_data_length = train_data_length
        self.validate_data_length = validate_data_length

    def parse_individual(self, indi):
        torch_device = torch.device('cuda')
        cnn = CNN(indi)
        cnn = cnn.cuda()
        print(cnn)
        complexity = get_total_params(cnn, (220, 30, 30))

        train_loader = get_data.get_train_loader(8)

        # Loss and optimizer 3.定义损失函数， 使用的是最小平方误差函数
        criterion = nn.MSELoss()
        criterion = criterion.to(torch_device)

        # 4.定义迭代优化算法， 使用的是随机梯度下降算法
        learning_rate = 0.001
        optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
        loss_dict = []
        num_epochs = train_loader.__len__()
        # Train the model 5. 迭代训练
        cnn.train()
        for i, data in enumerate(train_loader,0):
            # Convert numpy arrays to torch tensors  5.1 准备tensor的训练数据和标签
            inputs, labels = data
            labels = get_data.get_predict_size_labels(indi, labels)
            # labels = get_data.get_size_labels(indi.get_layer_size(),labels)

            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward pass  5.2 前向传播计算网络结构的输出结果
            optimizer.zero_grad()
            outputs = cnn(inputs)
            # 5.3 计算损失函数
            loss = criterion(outputs, labels)

            # Backward and optimize 5.4 反向传播更新参数
            loss.backward()
            optimizer.step()

            # 可选 5.5 打印训练信息和保存loss
            loss_dict.append(loss.item())
            if (i + 1) % 5 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(i + 1, num_epochs, loss.item()))

        # return mean_test_accu, np.std(test_accuracy_list), complexity, history_best_score


if __name__ == '__main__':
    import evolve

    cnn = evolve.Evolve_CNN(0.1, 10, 0.2, 1, 10, 1, 11, 1, 1, 1, 1, 1, 1, 1, 1)
    cnn.initialize_popualtion()

    evaluate = Evaluate(10, 1, 1, 1, 1, 220, 1, 32, 1000, 1000)
    evaluate.parse_individual(cnn.pops.get_individual_at(0))
