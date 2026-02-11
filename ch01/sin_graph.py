# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 创建数据
x = np.arange(0, 6, 0.1)  # 以0.1为单位，生成0到6的数据
y = np.sin(x)

# 绘制图形
plt.plot(x, y)

# 如果是服务器上，则显示不了图像，但可以保存下来
plt.savefig("ch01/sin_graph.png")  # ! 注意这里的路径是相对于终端运行时的路径

# plt.show()  # 在服务器上这一行通常不需要了
