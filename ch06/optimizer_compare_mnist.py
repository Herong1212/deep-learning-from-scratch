# coding: utf-8
import os
import sys

# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

# 将父目录的绝对路径加入系统环境变量，这样 Python 就能找到上一级文件夹里的 common 和 dataset 模块了
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# step0: MNIST数据集加载 ==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# step1: 实验设定 ==========
optimizers = {}
optimizers["SGD"] = SGD()
optimizers["Momentum"] = Momentum()
optimizers["AdaGrad"] = AdaGrad()
optimizers["Adam"] = Adam()
# optimizers['RMSprop'] = RMSprop()

networks = {}  # 存放 4 个神经网络模型
train_loss = {}  # 记录 4 个模型每一次迭代后的损失值，方便最后画折线图

# 遍历 4 个优化器的名字，为每一个名字分配一个专属的神经网络
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784,  # 输入层 784 个神经元
        hidden_size_list=[
            100,
            100,
            100,
            100,
        ],  # 非常核心！它是一个列表，里面有 4 个元素，代表我们要搭建 4 层隐藏层，而且每一层都有 100 个神经元。
        output_size=10,  # 输出层 10 个神经元，对应 0 到 9 这十个数字的预测概率
    )
    train_loss[key] = []


# step2: 训练开始 ==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)  # 随机选择 batch_size 个样本
    x_batch = x_train[batch_mask]  # shape= (128, 784)
    t_batch = t_train[batch_mask]  # shape= (128,)

    for key in optimizers.keys():
        grads = networks[key].gradient(
            x_batch, t_batch
        )  # 网络内部的高级方法（误差反向传播），可瞬间算出包含 5 层网络中所有 W1, b1, ..., W5, b5 数万个参数的梯度
        optimizers[key].update(
            networks[key].params, grads
        )  # 优化器拿着刚刚算出的梯度，去修改当前网络的底层参数

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(
            loss
        )  # 把更新后算出来的分数（越小越好）塞进列表里记账。经过 2000 次循环，每个列表里就会攒下 2000 个数字。

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# step3. 绘图 ==========
markers = {
    "SGD": "o",
    "Momentum": "x",
    "AdaGrad": "s",
    "Adam": "D",
}  # 为 4 条折线设定不同的点标记：圆圈、叉号、正方形、菱形
x = np.arange(max_iterations)  # 生成 0 到 1999 的数组，作为 X 轴坐标
for key in optimizers.keys():
    # * 工程小技巧。因为 batch_size 只有 128，每次抽到的数据有好有坏，原始的 Loss 折线图会像锯齿一样剧烈上下跳动，看不出趋势。
    # smooth_curve 函数会通过移动平均算法，把毛刺抹平，让曲线看起来丝滑连贯，方便我们对比大趋势。
    plt.plot(
        x,
        smooth_curve(train_loss[key]),  # Y轴数据
        marker=markers[key],  # 标记样式
        markevery=100,  # 每隔100个点才画一个标记（不然点太密集会糊成一团）
        label=key,  # 图例名字
    )
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()  # 显示图例框

# plt.show()
plt.savefig("ch06/optimizer_compare_mnist.png")
