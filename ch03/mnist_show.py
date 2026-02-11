# coding: utf-8
import sys, os

# sys.path：

# sys.path：这相当于 C++ 编译器里的 Include Path（头文件搜索路径，也就是 -I 参数）。
# 当执行 import dataset 时，Python 会遍历 sys.path 里的所有目录，看看哪个目录下面有 dataset 这个文件夹。
# append()：类似 std::vector::push_back，把一个新的路径加到搜索列表的末尾。
# os.pardir：全称是 Parent Directory。在 Linux/Mac 下，它就是一个字符串 ".."。
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    """
    显示图像函数。
    img (NumPy 数组)：只是一堆数字矩阵（比如 [[255, 0], [0, 255]]）。
                        它不知道这些数字代表“图片”，它只知道这是数学矩阵；没有 .show() 这种显示窗口的方法。
    pil_img (PIL 对象)：这就像一个封装好的图形对象，比如 MFC 的 CBitmap，Qt 的 QImage，或者 OpenCV 的 cv::Mat。
                        它拥有元数据（头部信息，如格式、尺寸、色彩模式），并且拥有成员函数（如 .show(), .save(), .resize()）。

    参数:
        img (numpy.ndarray): 输入的图像数据，通常为 NumPy 数组格式。

    功能说明:
        该函数将输入的图像数据转换为符合显示标准的格式，并使用 PIL 库显示图像。
        首先将图像数据强制转换为 8 位无符号整数（uint8），然后将其转换为 PIL 图像对象并显示。
    """
    # 强制类型转换，把数据强制变成 8 位无符号整数，符合图片格式标准。
    print(f"img:  \n", img)
    print(f"img type:  ", type(img))  # type:   <class 'numpy.ndarray'>
    print(f"img dtype:  ", img.dtype)  # img dtype:   uint8
    img_uint8 = np.uint8(img)
    print(f"img_uint8:  \n", img_uint8)
    print(f"img_uint8 type:  ", type(img_uint8))  # type:   <class 'numpy.ndarray'>
    print(f"img_uint8 dtype:  ", img_uint8.dtype)  # img_uint8 dtype:   uint8
    # fromarray() 把保存为 NumPy 数组的一堆数字矩阵转换为 PIL 用的具备成员属性/函数的图形对象
    # fromarray() 要求传入uint8 类型（0~255）的数组。如果是二维数组 (H, W)，它通常会被解析为灰度图。如果是三维数组 (H, W, 3)，它会被解析为 RGB 彩色图。
    # 参数2 mode: PIL 图像的格式，比如 "L" 表示灰度图（8位像素，黑白），如果未传入参数，且输入是二维数组，PIL 会默认猜这个。"RGB" 表示彩色图。"F"表示32位浮点像素（很少用，除非你在做科研数据可视化）。
    pil_img = Image.fromarray(img_uint8)
    print(f"pil_img:  \n", pil_img)
    pil_img.show()


# 第一次调用会花费几分钟……
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 输出各个数据的形状
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000,)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变成原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
