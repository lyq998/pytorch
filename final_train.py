import torch
from torch import nn
from cnn import CNN
from utils import *
import get_data
from tqdm import tqdm
from libtiff import TIFF


def train(model, batch_size, optimizer, model_type):
    torch_device = torch.device('cuda')
    model.cuda()
    if model_type == 1:
        train_loader = get_data.get_final_train_loader(batch_size)
    elif model_type == 2:
        train_loader = get_data.get_gauss50_final_train_loader(batch_size)
    else:
        train_loader = get_data.get_mixed_final_train_loader(batch_size)

    # Loss and optimizer 3.定义损失函数， 使用的是最小平方误差函数
    criterion = nn.MSELoss()
    criterion = criterion.to(torch_device)

    loss_dict = []
    num_epochs = train_loader.__len__()
    # Train the model 5. 迭代训练
    model.train()
    batch_tqdm = tqdm(enumerate(train_loader, 0), total=num_epochs, ncols=100)
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


def test(model, batch_size, model_type, save):
    # evaluate
    model.eval()
    torch_device = torch.device('cuda')
    # Loss and optimizer 定义损失函数， 使用的是最小平方误差函数
    criterion = nn.MSELoss()
    criterion = criterion.to(torch_device)
    eval_loss_dict = []

    if model_type == 1:
        test_loader = get_data.get_test_loader(batch_size)
    elif model_type == 2:
        test_loader = get_data.get_gauss50_test_loader(batch_size)
    else:
        test_loader = get_data.get_mixed_test_loader(batch_size)

    batch_tqdm = tqdm(enumerate(test_loader, 0), total=test_loader.__len__(), ncols=100)
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
    # 一定记住要在开始运行之前切换pops等数据
    model_type = 3  # 1:gauss200   2:gauss50   3:mixed
    test_flag = False
    save_flag = False
    batch_size = 10
    epoch = 0

    if model_type == 1:
        save_dir = os.getcwd() + '/model.pth'
    elif model_type == 2:
        save_dir = os.getcwd() + '/gauss50_model.pth'
    else:
        save_dir = os.getcwd() + '/mixed_model.pth'

    # 如果test_flag=True,则加载已保存的模型
    if test_flag:
        # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
        # 这里是从最后的pop.dat里读第一个indi
        _, pops, _ = load_population()
        indi = pops.pops[0]
        model = CNN(indi)
        # 一定要将模型先cuda
        model.cuda()
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['model'])
        test(model, batch_size, model_type, save_flag)
    else:
        # 这里是从最后的pop.dat里读第一个indi
        _, pops, _ = load_population()
        indi = pops.pops[0]
        model = CNN(indi)
        # 一定要将模型先cuda
        model.cuda()
        print('从pops中选择最好的个体保存模型')
        print(model)

        # 定义迭代优化算法， 使用的是Adam，SGD不行
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 如果有保存的模型，则加载模型，并在其基础上继续训练
        if os.path.exists(save_dir):
            checkpoint = torch.load(save_dir)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch'] + 1

        while (True):
            print('start epoch:{}'.format(epoch))
            train(model, batch_size, optimizer, model_type)
            # 保存模型
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, save_dir)
            epoch = epoch + 1
