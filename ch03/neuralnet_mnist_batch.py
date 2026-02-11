# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100  # 批次数量
accuracy_cnt = 0

# range() 函数若指定为 range(start, end)，则会生成一个由 start 到 end-1 之间的整数构成的列表。
# range(start, end, step) 这样指定 3 个整数，则生成的列表中的下一个元素会增加 step 指定的值。
for i in range(0, len(x), batch_size):
    # x[i:i+batch_n] 会取出从第 i 个到第 i+batch_n 个之间的数据
    x_batch = x[i : i + batch_size]  # 如 x[0:100]、x[100:200]……
    print(f"x_batch_{i}.shape = ", x_batch.shape)  # x_batch_9700.shape =  (100, 784)

    y_batch = predict(network, x_batch)
    print(f"y_batch_{i}.shape = ", y_batch.shape)  # y_batch_9700.shape =  (100, 10)

    # ! 矩阵的第 0 维是列方向，第 1 维是行方向。
    # 参数 axis=1，指定了在 100 × 10 的数组中，沿着第 1 维方向找到值最大的元素的索引（第 0 维对应第 1 个维度）
    # np.argmax(y_batch, axis=1) 的逻辑是： “请把每一行孤立地看，在每一行内部（横向）找到最大值的索引。”
    p = np.argmax(y_batch, axis=1)  # argmax() 获取值最大的元素的索引
    print(f"p_{i}.shape = ", p.shape)  # p_9700.shape =  (100,)
    print(f"p_{i} = ", p)
    # p_9700 = [6 9 5 2 0 1 2 3 4 5 6 7 5 9 0 1 0 3 4 0 6 7 8 9 0 1 0 3 4 6 6 7 5 9 7 4 6 1 6 0 9 7 3 7 1 2 7 5 8 6 3 0 0 0 5 8 6 0 3 8 1 0 3 0 4 7 4 9 0 9 0 7 1 7 1 6 6 0 6 0 8 7 6 4 9 9 5 3 7 4 3 0 9 6 6 1 1 3 2 1]

    accuracy_cnt += np.sum(p == t[i : i + batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
