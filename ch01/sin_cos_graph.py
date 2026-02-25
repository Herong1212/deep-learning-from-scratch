# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 创建数据
x = np.arange(0, 6, 0.1)  # 以0.1为单位，生成0到6的数据
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制图形
plt.plot(x, y1, label="sin")  # 设置了 label 后，必须调用 plt.legend() 才会显示图例
plt.plot(x, y2, linestyle="--", label="cos", color="r")  # 用虚线绘制
plt.xlabel("x")  # x轴的标签
plt.ylabel("y")  # y轴的标签
plt.title("sin & cos")  # 标题
plt.legend()  # 显示图例

# 同样保存到本地
plt.savefig("ch01/sin_cos_graph.png")

# plt.show()  # 服务器模式下无法显示图像
