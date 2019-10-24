# coding:utf-8
import numpy as np
from libtiff import TIFF

path0 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_EW-line_R.tif'
path1 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_NS-line.tif'
noise_path0 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.tif'
noise_path1 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_NS-line_Gaussnoise.tif'

train_image_path = 'G:/evocnn/image_set/image_set/train_image/images/'
train_label_path = 'G:/evocnn/image_set/image_set/train_image/labels/'
test_image_path = 'G:/evocnn/image_set/image_set/test_image/images/'
test_label_path = 'G:/evocnn/image_set/image_set/test_image/labels/'

imgdir0 = TIFF.open(path0, mode="r")
imgdir1 = TIFF.open(path1, mode="r")
imgdir3 = TIFF.open(noise_path0, mode="r")
imgdir4 = TIFF.open(noise_path1, mode="r")

labarr0 = imgdir0.read_image()
labarr1 = imgdir1.read_image()
imgarr0 = imgdir3.read_image()
imgarr1 = imgdir4.read_image()


def copy_img(row, col, imgarr):
    for k in range(220):
        for i in range(30):
            for j in range(30):
                save_arr[k][i][j] = imgarr[k][i + row][j + col]
    return save_arr

if __name__ == '__main__':
    '''
    因为是要分割成30*30的图像
    i,j的循环次数要自己计算出边界
    分别是：i           j
            59          182
            265         59
    将第一张图分成两份，前59*82作为test_image
    后面59*100的加上图二一起作为train_image
    '''
    save_arr = np.zeros((220, 30, 30), dtype=np.uint16)
    # save test_image
    print('start creat test_image:')
    index = 0
    for i in range(59):
        for j in range(82):
            save_arr = copy_img(i * 10, j * 10, imgarr0)
            save_imgname = test_image_path + str(index) + '.tif'
            img = TIFF.open(save_imgname, 'w')
            img.write_image(save_arr, write_rgb=True)

            save_arr = copy_img(i * 10, j * 10, labarr0)
            save_imgname = test_label_path + str(index) + '.tif'
            img = TIFF.open(save_imgname, 'w')
            img.write_image(save_arr, write_rgb=True)

            index = index + 1
            if index % 100 == 0:
                print('{}     success!'.format(index))

    # save train_image
    print('start creat train_image:')
    index = 0
    for i in range(59):
        for j in range(100):
            save_arr = copy_img(i * 10, (j + 82) * 10, imgarr0)
            save_imgname = train_image_path + str(index) + '.tif'
            img = TIFF.open(save_imgname, 'w')
            img.write_image(save_arr, write_rgb=True)

            save_arr = copy_img(i * 10, (j + 82) * 10, labarr0)
            save_imgname = train_label_path + str(index) + '.tif'
            img = TIFF.open(save_imgname, 'w')
            img.write_image(save_arr, write_rgb=True)

            index = index + 1
            if index % 100 == 0:
                print('{}     success!'.format(index))

    index = 5900
    # 当为图一加图二时，初值为5900
    for i in range(265):
        for j in range(59):
            save_arr = copy_img(i * 10, j * 10, imgarr1)
            save_imgname = train_image_path + str(index) + '.tif'
            img = TIFF.open(save_imgname, 'w')
            img.write_image(save_arr, write_rgb=True)

            save_arr = copy_img(i * 10, j * 10, labarr1)
            save_imgname = train_label_path + str(index) + '.tif'
            img = TIFF.open(save_imgname, 'w')
            img.write_image(save_arr, write_rgb=True)

            index = index + 1
            if index % 100 == 0:
                print('%d     success!' % index)

