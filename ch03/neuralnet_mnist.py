# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np

# Python 标准库，用于 序列化 (Serialization)
# 类似于 Boost.Serialization 或者把结构体直接 fwrite 进二进制文件。它可以把内存里的对象（字典、数组、类实例）原封不动地保存到硬盘上，以后再读回来。
import pickle

from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )

    # 因为当前脚本的任务是“推理”（Inference/Test），而不是“训练”（Training），所以不需要返回 x_train, t_train
    return x_test, t_test


def init_network():
    # 加载保存过的 pickle 文件，可以立刻复原之前程序运行中的对象。
    # 这个文件中以字典变量的形式保存了权重和偏置参数。
    with open("sample_weight.pkl", "rb") as f:
        # with：上下文管理器，当离开 with 代码块（缩进结束）时，Python 会自动调用 f.close()。哪怕中间报错了，文件也会被安全关闭。就不用担心忘记写 fclose() 了。
        # "rb"：Read (只读) + Binary (二进制模式)。为什么必须是 b？ 因为 pickle 保存的是字节流，不是文本。如果用普通文本模式打开，换行符转换会破坏数据。
        # as f: 给打开的文件句柄起个别名，叫 f (file handle)。
        print(f.raw)  # <_io.FileIO name='sample_weight.pkl' mode='rb' closefd=True>
        network = pickle.load(f)

        # --- 调试代码开始 ---
        print("Keys:", network.keys())  # 看看有哪些层
        # Keys: dict_keys(["b2", "W1", "b1", "W2", "W3", "b3"])

        # 看看每一层权重的形状（重要！）
        for key in network.keys():
            print(f"{key} shape = : {network[key].shape}")
            print(f"{key} type = : {type(network[key])}")
            print(f"{key} dtype = : {network[key].dtype}")
            # b2 shape = : (100,)
            # b2 type =  : <class 'numpy.ndarray'>
            # b2 dtype = : float32
            # W1 shape = : (784, 50)
            # W1 type =  : <class 'numpy.ndarray'>
            # W1 dtype = : float32
            # b1 shape = : (50,)
            # b1 type =  : <class 'numpy.ndarray'>
            # b1 dtype = : float32
            # W2 shape = : (50, 100)
            # W2 type =  : <class 'numpy.ndarray'>
            # W2 dtype = : float32
            # W3 shape = : (100, 10)
            # W3 type =  : <class 'numpy.ndarray'>
            # W3 dtype = : float32
            # b3 shape = : (10,)
            # b3 type =  : <class 'numpy.ndarray'>
            # b3 dtype = : float32

        # --- 调试代码结束 ---

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    np.set_printoptions(suppress=True, precision=6)  # * 禁用科学计数法，保留6位小数

    return y


x, t = get_data()
print(f"x:  \n", x)
print(f"x.shape:  ", x.shape)  # x.shape:   (10000, 784)
# x:
#     [
#         [0. 0. 0. ... 0. 0. 0.]
#         [0. 0. 0. ... 0. 0. 0.]
#         [0. 0. 0. ... 0. 0. 0.]
#         ...
#         [0. 0. 0. ... 0. 0. 0.]
#         [0. 0. 0. ... 0. 0. 0.]
#         [0. 0. 0. ... 0. 0. 0.]
#     ]
print(f"x[0].shape = ", x[0].shape)  # x[0].shape =  (784,)
print(f"t:  ", t)  # t:   [7 2 1 ... 4 5 6]
print(f"t.type:  ", type(t))  # t.type:   <class 'numpy.ndarray'>
print(f"t.length:  ", len(t))  # t.length:   10000

network = init_network()
# print(f"network:  \n", network)
print(f"network type:  ", type(network))  # network type:   <class 'dict'>

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    print(f"y_{i} = ", y)
    # y_9986 = [0.00004  0.013202 0.010458 0.399174 0.000017 0.020913 0.000006 0.003422 0.546987 0.00578]
    print(f"y_{i} type = ", type(y))
    # y_9986 type:   <class 'numpy.ndarray'>
    print(f"y_{i}.shape = ", y.shape)
    # y_9986.shape =  (10,)

    p = np.argmax(y)  # 获取概率最高的元素的索引
    print(f"p_{i} = ", p)  # p_9986 =  8

    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
