# coding:utf-8
import numpy as np
from libtiff import TIFF
from tqdm import tqdm
import random

path0 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_EW-line_R.tif'
path1 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_NS-line.tif'
guass50_path0 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise_50.tif'
guass50_path1 = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_NS-line_Gaussnoise_50.tif'
# imgdir0 = TIFF.open(path0, mode="r")
imgdir1 = TIFF.open(path1, mode="r")

# imgarr0 = imgdir0.read_image()
imgarr1 = imgdir1.read_image()
# imgarr2 = TIFF.open(guass50_path0, mode="r").read_image()

# imgarr的shape[0][1][2]分别对应通道，宽，长，反过来的
# noise_add0 = np.random.normal(size=(220, 614, 1848))
# noise_add1 = np.random.normal(size=(220, 2678, 614))
# np.random.normal()只能创建二维正态分布

# 加高斯噪声的代码
for i in tqdm(range(220)):
    # noise_add0 = np.random.normal(size=(614, 1848))
    noise_add1 = np.random.normal(size=(2678, 614))
    for j in range(2678):
        for k in range(614):
            imgarr1[i][j][k] += noise_add1[j][k] * 50
            if(imgarr1[i][j][k] > 65535):
                imgarr1[i][j][k] = 65535
            elif(imgarr1[i][j][k] < 0):
                imgarr1[i][j][k] = 0


# 加条带噪声
# for i in tqdm(range(220)):
#     if i % 20 == 0:
#         num = range(0, 614)
#         nums = random.sample(num, 123)  # 选取x个元素
#         print(nums)
#         for iii in range(len(nums) - 1):
#             for jjj in range(len(nums) - iii - 1):
#                 if nums[jjj] > nums[jjj + 1]:
#                     nums[jjj], nums[jjj + 1] = nums[jjj + 1], nums[jjj]
#         print(nums)
#
#         for j in range(2678):
#             for k in nums:
#                 imgarr1[i][j][k] = 0

imgname = 'G:/evocnn/image_set/original_image/19920612_AVIRIS_IndianPine_NS-line_mixed.tif'
img = TIFF.open(guass50_path1, 'w')
img.write_image(imgarr1, write_rgb=True)
print('success!')
