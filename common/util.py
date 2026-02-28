# coding: utf-8
import numpy as np


def smooth_curve(x):
    """
    对输入数据进行平滑处理，通常用于平滑损失函数的曲线图。

    参数:
        x (array-like): 需要平滑处理的一维数组或列表。

    返回:
        array: 平滑处理后的数组，长度比输入数组略短。

    参考: http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    # 设置窗口长度，用于定义平滑的范围
    window_len = 11

    # 扩展输入数组，以处理边界效应
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]

    # 生成Kaiser窗函数，用于加权平均
    w = np.kaiser(window_len, 2)

    # 使用卷积操作对扩展后的数组进行平滑处理
    y = np.convolve(w / w.sum(), s, mode="valid")

    # 去除由于卷积引入的额外边界点，返回中心部分的结果
    return y[5 : len(y) - 5]


def shuffle_dataset(x, t):
    """对数据集进行随机打乱操作

    该函数通过生成一个随机排列索引，对训练数据和对应的教师数据进行同步打乱，
    确保数据顺序的随机性，同时保持训练数据与教师数据的一一对应关系。

    Parameters
    ----------
    x : numpy.ndarray
        训练数据，形状为 (样本数, 特征数) 或 (样本数, 高, 宽, 通道数)
    t : numpy.ndarray
        教师数据（标签），形状为 (样本数,)

    Returns
    -------
    x : numpy.ndarray
        打乱后的训练数据，形状与输入相同
    t : numpy.ndarray
        打乱后的教师数据，形状与输入相同
    """
    # 生成一个长度为样本数的随机排列索引
    permutation = np.random.permutation(x.shape[0])

    # 根据索引对训练数据进行打乱，支持二维和四维数据
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]

    # 根据索引对教师数据进行打乱
    t = t[permutation]

    return x, t


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2 * pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将输入数据转换为列格式，用于卷积操作的高效计算。

    参数:
    input_data : numpy.ndarray, 形状为 (数据数, 通道数, 高度, 宽度) 的四维输入数据。
    filter_h : int, 卷积核的高度。
    filter_w : int, 卷积核的宽度。
    stride : int, 可选, 卷积操作的步幅，默认为 1。
    pad : int, 可选, 输入数据的零填充大小，默认为 0。

    返回:
    col : numpy.ndarray, 转换后的二维数组，形状为 (数据数 * 输出高度 * 输出宽度, 通道数 * 卷积核高度 * 卷积核宽度)。
    """
    # 获取输入数据的形状
    N, C, H, W = input_data.shape

    # 计算输出特征图的高度和宽度
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    # print(f"out_h: {out_h}, out_w: {out_w}")

    # 对输入数据进行零填充（这行代码执行的就是纯正的 Zero Padding（零填充））
    # * 在卷积神经网络（CNN）的图像分类任务中，几乎 100% 只用 0 来填充（Zero Padding）。
    # 维度对应关系拆解：
    #   第 0 维 (N, 批处理/图片数量)：对应第 1 个元组 (0, 0)。表示在第一张图片前面和最后一张图片后面，不增加任何新图片。
    #   第 1 维 (C, 通道数)：对应第 2 个元组 (0, 0)。表示在第一个通道前和最后一个通道后，不增加任何新通道。
    #   第 2 维 (H, 高度)：对应第 3 个元组 (pad, pad)。表示在图像的最上方填充 pad 行，在最下方填充 pad 行。
    #   第 3 维 (W, 宽度)：对应第 4 个元组 (pad, pad)。表示在图像的最左侧填充 pad 列，在最右侧填充 pad 列。
    img = np.pad(
        input_data,
        [(0, 0), (0, 0), (pad, pad), (pad, pad)],
        "constant",  # 指定填充方式为用常量值进行填充，默认为 constant
        constant_values=0,  # 指定要填充的常量数值，默认为 0
    )

    # 在内存中初始化一个全是 0 的 6 维空数组，作为后续存放提取出来的像素点的“临时中转站（模具）”
    col = np.zeros(
        (
            N,  # 批处理大小（Batch Size），即一共有几张图片。
            C,  # 通道数（Channel），即每张图片有几个图层（如 RGB 为 3）。
            filter_h,  # 滤波器（卷积核）的高度。
            filter_w,  # 滤波器（卷积核）的宽度。
            out_h,  # 输出特征图的高度（即滤波器在垂直方向上总共会滑动/停顿多少次）。
            out_w,  # 输出特征图的宽度（即滤波器在水平方向上总共会滑动/停顿多少次）。
        )
    )

    # 遍历卷积核的每个位置，提取对应的图像区域
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 调整数组维度并重塑为二维形式
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad : H + pad, pad : W + pad]
