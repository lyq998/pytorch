from tqdm import tqdm
from libtiff import TIFF
import numpy as np

save_arr = np.zeros((220, 608, 838), dtype=np.uint16)
def copy_img(row, col, imgarr):
    for k in range(220):
        for i in range(28):
            for j in range(28):
                if save_arr[k][i + row][j + col] == 0:
                    save_arr[k][i + row][j + col] = imgarr[k][i][j]
    return save_arr

index = 0
# 因为读取的是反的：（通道*宽*长）
img = np.zeros((220, 608, 838), dtype=np.uint16)
# output_img = np.zeros((220, 608, 838), dtype=np.uint16)
count = np.zeros((608, 838), dtype=np.uint16)
for i in tqdm(range(59)):
    for j in range(82):
        patch = TIFF.open('G:/evocnn/image_set/output/' + str(index) + '.tif', mode="r").read_image()
        copy_img(i * 10, j * 10, patch)
        # img[:, j * 10:j * 10 + 28, i * 10:i * 10 + 28] += patch.astype('uint16')
        # count[i * 10:i * 10 + 28, j * 10:j * 10 + 28] += 1
        index += 1

# for i in tqdm(range(608)):
#     for j in range(838):
#         save_arr[:,i, j] = (save_arr[:,i, j] / count[i, j]).astype('uint16')

# save img
# for  k in range(220):
#     output_img[k] = np.transpose(img[k])
save_img = TIFF.open('G:/evocnn/image_set/output/new_combine.tif', mode="w")
save_img.write_image(save_arr, write_rgb=True)
