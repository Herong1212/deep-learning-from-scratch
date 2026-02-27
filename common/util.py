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
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    
    # 生成Kaiser窗函数，用于加权平均
    w = np.kaiser(window_len, 2)
    
    # 使用卷积操作对扩展后的数组进行平滑处理
    y = np.convolve(w/w.sum(), s, mode='valid')
    
    # 去除由于卷积引入的额外边界点，返回中心部分的结果
    return y[5:len(y)-5]


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
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    
    # 根据索引对教师数据进行打乱
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
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
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]