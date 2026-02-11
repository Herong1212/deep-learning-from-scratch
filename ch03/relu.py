# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


# ReLU 函数在输入大于 0 时，直接输出该值；在输入小于等于 0 时，输出0
def relu(x):
    # maximum 函数会从输入的数值中选择较大的那个值进行输出
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
# plt.show()
plt.title("ReLU Function")
plt.savefig("relu.png")
