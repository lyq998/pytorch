from torch import nn
import torch

a=[1,2,3,4,5,6]
b=[9,9,9]
a[3:6]=b
a.extend(b)
print(a)