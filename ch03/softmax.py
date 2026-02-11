import numpy as np

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)  # 指数函数的和
print(sum_exp_a)

a_out = exp_a / sum_exp_a
print(a_out)


def softmax(a):
    """
    计算输入数组的softmax值。
    输出满足:
            1、每个 y_i 都在 0~1 之间;
            2、所有 y_i 加起来等于 1

    参数:
        a (numpy.ndarray): 输入的数值数组，通常为一维或二维数组。

    返回:
        numpy.ndarray: 经过softmax变换后的数组，每个元素表示对应输入元素的概率分布。
    """

    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# ! 存在缺陷：因为要进行指数运算，可能会导致数值溢出。
b = np.array([1010, 1000, 990])
b_out = np.exp(b) / np.sum(np.exp(b))  # softmax函数的运算
print(b_out)  # [nan nan nan]，说明没有被正确计算

# * 解决方案： 数学上可以证明 ———— Softmax 的结果不受所有输入同时加上或减去一个常数的影响。
# 具体实现：先把输入向量里的最大值 b_max 找出来，然后所有数 b_i 都减去这个最大值，再进行指数运算。
b_max = np.max(b)  # b_max = 1010
print(b - b_max)  # [  0 -10 -20]
b_out1 = np.exp(b - b_max) / np.sum(np.exp(b - b_max))
print(b_out1)  # [9.99954600e-01 4.53978686e-05 2.06106005e-09]


# note 修改版本👇
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


d = np.array([0.3, 2.9, 4.0])
e = softmax(d)
print(e)  # [0.01821127 0.24519181 0.73659691]
print(np.sum(e)) # 1.0 
