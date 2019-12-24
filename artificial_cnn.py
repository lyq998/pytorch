import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import get_data
from tqdm import tqdm
import numpy as np
from libtiff import TIFF
from torchsummary import summary


class ArtificialCNN(nn.Module):
    def __init__(self):
        super(ArtificialCNN, self).__init__()
        self.conv1 = nn.Conv2d(220, 220, 3, 1, 1)
        self.conv2 = nn.Conv2d(220, 220, 3, 1, 1)
        self.conv3 = nn.Conv2d(220, 220, 3, 1, 1)
        self.conv4 = nn.Conv2d(220, 220, 3, 1, 1)
        self.conv5 = nn.Conv2d(220, 440, 3, 1, 1)
        self.conv6 = nn.Conv2d(440, 440, 3, 1, 1)
        self.conv7 = nn.Conv2d(440, 440, 3, 1, 1)
        self.conv8 = nn.Conv2d(440, 880, 3, 1, 1)
        self.conv9 = nn.Conv2d(880, 880, 3, 1, 1)
        self.conv10 = nn.Conv2d(880, 880, 3, 1, 1)
        self.conv11 = nn.Conv2d(880, 440, 3, 1, 1)
        self.conv12 = nn.Conv2d(440, 440, 3, 1, 1)
        self.conv13 = nn.Conv2d(440, 440, 3, 1, 1)
        self.conv14 = nn.Conv2d(440, 220, 3, 1, 1)
        self.conv15 = nn.Conv2d(220, 220, 3, 1, 1)
        self.conv16 = nn.Conv2d(220, 220, 3)

        self.bn1 = nn.BatchNorm2d(220)
        self.bn2 = nn.BatchNorm2d(220)
        self.bn3 = nn.BatchNorm2d(220)
        self.bn4 = nn.BatchNorm2d(220)
        self.bn5 = nn.BatchNorm2d(440)
        self.bn6 = nn.BatchNorm2d(440)
        self.bn7 = nn.BatchNorm2d(440)
        self.bn8 = nn.BatchNorm2d(880)
        self.bn9 = nn.BatchNorm2d(880)
        self.bn10 = nn.BatchNorm2d(880)
        self.bn11 = nn.BatchNorm2d(440)
        self.bn12 = nn.BatchNorm2d(440)
        self.bn13 = nn.BatchNorm2d(440)
        self.bn14 = nn.BatchNorm2d(220)
        self.bn15 = nn.BatchNorm2d(220)

    def _initialize_weights(self):
        # print(self.modules())
        for m in self.modules():
            # print(m)
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                # print(m.weight)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = self.conv16(x)
        return x

def train(model, batch_size, optimizer):
    torch_device = torch.device('cuda')
    model.cuda()
    train_loader = get_data.get_final_train_loader(batch_size)

    # Loss and optimizer 3.定义损失函数， 使用的是最小平方误差函数
    criterion = nn.MSELoss()
    criterion = criterion.to(torch_device)

    loss_dict = []
    num_epochs = train_loader.__len__()
    # Train the model 5. 迭代训练
    model.train()
    batch_tqdm = tqdm(enumerate(train_loader, 0), total=num_epochs)
    for i, data in batch_tqdm:
        # Convert numpy arrays to torch tensors  5.1 准备tensor的训练数据和标签
        inputs, labels = data
        labels = get_data.get_size_labels(1, labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        # labels = get_data.get_size_labels(indi.get_layer_size(),labels)

        # Forward pass  5.2 前向传播计算网络结构的输出结果
        optimizer.zero_grad()
        outputs = model(inputs)
        # 5.3 计算损失函数
        loss = criterion(outputs, labels)
        loss = loss.cuda()

        # Backward and optimize 5.4 反向传播更新参数
        loss.backward()
        optimizer.step()

        # 可选 5.5 打印训练信息和保存loss
        loss_dict.append(loss.item())
    print('Loss: {:.4f}'.format(np.mean(loss_dict)))
    file_path = os.getcwd() + '/loss.txt'
    with open(file_path, 'a') as myfile:
        myfile.write(str(np.mean(loss_dict)))
        myfile.write("\n")


def test(model, batch_size, save):
    # evaluate
    model.eval()
    torch_device = torch.device('cuda')
    # Loss and optimizer 定义损失函数， 使用的是最小平方误差函数
    criterion = nn.MSELoss()
    criterion = criterion.to(torch_device)
    eval_loss_dict = []

    test_loader = get_data.get_test_loader(batch_size)

    batch_tqdm = tqdm(enumerate(test_loader, 0), total=test_loader.__len__())
    for i, data in batch_tqdm:
        inputs, labels = data
        labels = get_data.get_size_labels(1, labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss.cuda()
        eval_loss_dict.append(loss.item())

        if save:
            for j in range(10):
                save_imgname = 'G:/evocnn/image_set/output/' + str(i * 10 + j) + '.tif'
                img = TIFF.open(save_imgname, 'w')
                # 要保存之前先要改成CPU
                output_img = outputs[j].cpu().detach().numpy()
                img.write_image(output_img, write_rgb=True)

    mean_test_loss = np.mean(eval_loss_dict)
    std_test_loss = np.std(eval_loss_dict)
    print("valid mean:{},std:{}".format(mean_test_loss, std_test_loss))

if __name__ == '__main__':
    save_dir = os.getcwd() + '/artificialmodel.pth'
    batch_size = 128
    test_flag = False
    save_flag = False
    epoch = 0

    model = ArtificialCNN()
    model._initialize_weights()
    model.cuda()

    summary(model,(220,30,30))
    # learning_rate = 0.001
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #
    # # 如果test_flag=True,则加载已保存的模型
    # if test_flag:
    #     # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
    #     # 一定要将模型先cuda
    #     model.cuda()
    #     checkpoint = torch.load(save_dir)
    #     model.load_state_dict(checkpoint['model'])
    #     test(model, batch_size, save_flag)
    # else:
    #     # 一定要将模型先cuda
    #     model.cuda()
    #     print(model)
    #
    #     if os.path.exists(save_dir):
    #         checkpoint = torch.load(save_dir)
    #         model.load_state_dict(checkpoint['model'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         epoch = checkpoint['epoch'] + 1
    #
    #     while (True):
    #         print('start epoch:{}'.format(epoch))
    #         train(model, batch_size, optimizer)
    #         # 保存模型
    #         state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    #         torch.save(state, save_dir)
    #         epoch = epoch + 1

