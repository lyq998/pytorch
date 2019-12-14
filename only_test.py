import numpy as np
import os

image_root = 'G:/evocnn/image_set/image_set/test_image/images/'
image_files = np.array([x.path for x in os.scandir(image_root) if
                        x.name.endswith(".tif")])
print(image_files)

new_image_files = np.array([image_root + str(i) + '.tif' for i in range(4838)])
# for i in range(4838):
#     # new_image_files[i] = image_root + str(i) + '.tif'
#     new_image_files.extend(image_root + str(i) + '.tif')

print('')
