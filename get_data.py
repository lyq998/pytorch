from torch.utils.data import Dataset, DataLoader, TensorDataset
from libtiff import TIFF
from torch.autograd import Variable
import numpy as np
import os
import torch

train_image_path = 'G:/evocnn/image_set/image_set/train_image/images/'
train_label_path = 'G:/evocnn/image_set/image_set/train_image/labels/'
test_image_path = 'G:/evocnn/image_set/image_set/test_image/images/'
test_label_path = 'G:/evocnn/image_set/image_set/test_image/labels/'
validate_image_path = 'G:/evocnn/image_set/image_set/train_image/images/validate_image/'
validate_label_path = 'G:/evocnn/image_set/image_set/train_image/labels/validate_label/'


class TiffDataset(Dataset):
    def __init__(self, image_root, label_root):
        # 这个list存放所有图像的地址
        self.image_files = np.array([x.path for x in os.scandir(image_root) if
                                     x.name.endswith(".tif")])
        # label
        self.label_files = np.array([x.path for x in os.scandir(label_root) if
                                     x.name.endswith(".tif")])

    def __getitem__(self, index):
        # 读取图像数据并返回，返回训练image以及对应的label
        # 注意：只能返回float32，进行训练
        return torch.from_numpy(TIFF.open(self.image_files[index]).read_image().astype('float32')), torch.from_numpy(
            TIFF.open(self.label_files[index]).read_image().astype('float32'))

    def __len__(self):
        # 返回图像的数量
        return len(self.image_files)

    def get_image_files_at(self, index):
        return self.image_files[index]


def get_train_loader(batch_size):
    image_dataset = TiffDataset(train_image_path, train_label_path)
    train_loader = DataLoader(dataset=image_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    # 因为这是为了训练给evolve的初次选择所以shuffle = False
    return train_loader


def get_test_loader(batch_size):
    image_dataset = TiffDataset(test_image_path, test_label_path)
    test_loader = DataLoader(dataset=image_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    return test_loader


def get_validate_loader(batch_size):
    image_dataset = TiffDataset(validate_image_path, validate_label_path)
    validate_loader = DataLoader(dataset=image_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    return validate_loader


def get_predict_size_labels(indi, labels):
    num_of_units = indi.get_layer_size()
    count_of_size3 = 0
    for i in range(num_of_units):
        current_unit = indi.get_layer_at(i)
        if current_unit.type == 1:
            if current_unit.filter_height == 3:
                # print('filter_size_3')
                count_of_size3 += 1
        elif current_unit.type == 2:
            pass
        else:
            raise NameError('No unit with type value {}'.format(current_unit.type))
    if count_of_size3 == 0:
        return labels
    else:
        labels = labels[:,:,count_of_size3:(-count_of_size3),count_of_size3:(-count_of_size3)]
        return labels

def get_size_labels(num,lables):
    return lables[:,:,num:-num,num:-num]


if __name__ == '__main__':
    train_loader = get_train_loader(32)
    print('train_lodaer_len:', train_loader.__len__())
    print('dataset_len:', train_loader.dataset.__len__())
    torch_device = torch.device('cuda')
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(torch_device)
        print(inputs.dtype)
    # for epoch in range(2):
    #     for i, data in enumerate(train_loader):
    #         inputs, labels = data
    #         # inputs,labels=Variable(inputs),Variable(labels)
    #         print(epoch, i, 'inputs', inputs.data.size(), 'labels', labels.data.size())
