# coding:utf-8
import numpy as np
from libtiff import TIFF
from tqdm import tqdm

path0 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_EW-line_R.tif'
path1 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_NS-line.tif'
noise_path0 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.tif'
noise_path1 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_NS-line_Gaussnoise.tif'
gauss50_noise_path0 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise_50.tif'
gauss50_noise_path1 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_NS-line_Gaussnoise_50.tif'
mixed_noise_path0 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_EW-line_R_mixed.tif'
mixed_noise_path1 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_NS-line_mixed.tif'

train_image_path = 'G:/evocnn/image_set/image_set/train_image/images/'
train_label_path = 'G:/evocnn/image_set/image_set/train_image/labels/'
test_image_path = 'G:/evocnn/image_set/image_set/test_image/images/'
test_label_path = 'G:/evocnn/image_set/image_set/test_image/labels/'
mixed_test_image_path = 'G:/evocnn/image_set/image_set/mixed_test_image/images/'
mixed_train_image_path = 'G:/evocnn/image_set/image_set/mixed_train_image/images/'
mixed_train_label_path = 'G:/evocnn/image_set/image_set/mixed_train_image/labels/'

train_mixed_image_path = 'G:/evocnn/image_set/image_set/mixed_train_image/images/'
train_mixed_label_path = 'G:/evocnn/image_set/image_set/mixed_train_image/labels/'

gauss50_noise_test_image_path = 'G:/evocnn/image_set/image_set/gauss50_test_image/images/'
gauss50_noise_train_image_path = 'G:/evocnn/image_set/image_set/final_train_image/guass50_images/'

imgdir0 = TIFF.open(path0, mode="r")
imgdir1 = TIFF.open(path1, mode="r")
imgdir3 = TIFF.open(noise_path0, mode="r")
imgdir4 = TIFF.open(noise_path1, mode="r")
imgdir5 = TIFF.open(gauss50_noise_path0, mode="r")
imgdir51 = TIFF.open(gauss50_noise_path1, mode="r")
imgdir6 = TIFF.open(mixed_noise_path0, mode="r")
imgdir7 = TIFF.open(mixed_noise_path1, mode="r")

labarr0 = imgdir0.read_image()
labarr1 = imgdir1.read_image()
imgarr0 = imgdir3.read_image()
imgarr1 = imgdir4.read_image()
imgarr2 = imgdir5.read_image()
imgarr51 = imgdir51.read_image()
imgarr3 = imgdir6.read_image()
imgarr4 = imgdir7.read_image()


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
    # print('start creat test_image:')
    # index = 0
    # for i in tqdm(range(59)):
    #     for j in range(82):
    #         save_arr = copy_img(i * 10, j * 10, imgarr3)
    #         save_imgname = mixed_test_image_path + str(index) + '.tif'
    #         img = TIFF.open(save_imgname, 'w')
    #         img.write_image(save_arr, write_rgb=True)
    #
    #         # save_arr = copy_img(i * 10, j * 10, labarr0)
    #         # save_imgname = test_label_path + str(index) + '.tif'
    #         # img = TIFF.open(save_imgname, 'w')
    #         # img.write_image(save_arr, write_rgb=True)
    #
    #         index = index + 1

    # save train_image
    print('start creat train_image:')
    index = 0
    for i in tqdm(range(59)):
        for j in range(100):
            save_arr = copy_img(i * 10, (j + 82) * 10, imgarr2)
            save_imgname = gauss50_noise_train_image_path + str(index) + '.tif'
            img = TIFF.open(save_imgname, 'w')
            img.write_image(save_arr, write_rgb=True)

            # save_arr = copy_img(i * 10, (j + 82) * 10, labarr0)
            # save_imgname = train_label_path + str(index) + '.tif'
            # img = TIFF.open(save_imgname, 'w')
            # img.write_image(save_arr, write_rgb=True)

            index = index + 1

    index = 5900
    # 当为图一加图二时，初值为5900
    for i in tqdm(range(265)):
        for j in range(59):
            save_arr = copy_img(i * 10, j * 10, imgarr51)
            save_imgname = gauss50_noise_train_image_path + str(index) + '.tif'
            img = TIFF.open(save_imgname, 'w')
            img.write_image(save_arr, write_rgb=True)

            # save_arr = copy_img(i * 10, j * 10, labarr1)
            # save_imgname = train_label_path + str(index) + '.tif'
            # img = TIFF.open(save_imgname, 'w')
            # img.write_image(save_arr, write_rgb=True)

            index = index + 1

