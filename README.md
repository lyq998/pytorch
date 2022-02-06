# pytorch
The code for paper: Yuqiao Liu, Yanan Sun, Bing Xue, and Mengjie Zhang. "Evolving Deep Convolutional Neural Networks for Hyperspectral Image Denoising."
论文所使用的python代码，此部分代码可以分为四大类：
## 1. 数据集的构建：
   1. 加模拟噪声：add_noise.py
   2. 分割图像成小块儿：make_trainAndtest_image_set.py
   
之后的 train set and validation set 数据集搭建要靠自己拖动图片放到相应的文件夹

## 2. 演化过程的实现：
   1. 演化算法的主函数： main.py
   2. 种群类： population.py
   3. 个体类： individual.py
   4. 网络中的层类： layer.py
   5. 根据个体携带的编码搭建对应的神经网络： cnn.py
   6. 进化过程：evolve.py（包括了初始化种群，适应度评价，crossover and mutation，以及环境选择操作）
   7. 适应度评价的实现代码： evaluate.py

## 3. 最终训练和对比：
   1. 对选择出来的模型进行训练以及得到训练后网络的输出图像：final_train.py
   2. 人工搭建的神经网络：artificial_cnn.py
   
## 4. 工具包：
   1. utils.py
   2. 用来计算神经网络参数个数： nn_summery.py
   3. 获取训练数据： getdata.py
