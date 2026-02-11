import numpy as np


# 0. 定义激活函数 (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 0. 定义恒等函数 (输出层用)
# * 一般的，回归问题——恒等函数（输入与输出相同），二分类问题——Sigmoid 函数，多分类问题——Softmax 函数
def identity_function(x):
    # 不需要特殊处理，算出多少就是多少。比如输出 35.8，就是预测房价 35.8 万。
    return x


# 1. 初始化网络 (类似于 C++ 的构造函数 / Init)
def init_network():

    network = {}
    # 第一层权重：输入(2) -> 输出(3)
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # Shape: (2, 3)
    network["b1"] = np.array([0.1, 0.2, 0.3])  # Shape: (3,)

    # 第二层权重：输入(3) -> 输出(2)
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])  # Shape: (3, 2)
    network["b2"] = np.array([0.1, 0.2])  # Shape: (2,)

    # 第三层权重：输入(2) -> 输出(2)
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])  # Shape: (2, 2)
    network["b3"] = np.array([0.1, 0.2])  # Shape: (2,)
    return network


# 2. 前向传播 (核心逻辑)
def forward(network, x):
    # 取出参数 (类似于从 struct 里取成员)
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    # --- Layer 1 ---
    a1 = np.dot(x, W1) + b1  # 线性运算 (Affine)
    z1 = sigmoid(a1)  # 非线性激活

    # --- Layer 2 ---
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    # --- Layer 3 (输出层) ---
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)  # 如果是分类问题，这里通常换成 softmax

    return y


# 3. 主函数
network = init_network()
x = np.array([1.0, 0.5])  # ! 输入数据 Shape: (2,) -> 会被视为 (1, 2)
y = forward(network, x)
print(y)  # [ 0.31682708 0.69627909]
