# coding: utf-8
import numpy as np


def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


def numerical_gradient(f, x):
    """
    计算函数f在点x处的数值梯度。

    参数:
        f: 可调用对象，表示需要计算梯度的目标函数。
        x: numpy 数组，表示函数 f 的输入变量点。

    返回值:
        grad: numpy 数组，表示函数 f 在点 x 处的梯度。
    """
    h = 1e-4  # 0.0001，设置微小增量 h 用于数值微分
    grad = np.zeros_like(x)  # 初始化梯度数组，形状与 x 相同

    # 使用 numpy 的迭代器遍历 x 的所有元素
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index  # 获取当前元素的索引
        tmp_val = x[idx]  # 保存当前元素的原始值
        x[idx] = tmp_val + h  # 将当前元素增加h
        fxh1 = f(x)  # 计算 f(x+h)

        x[idx] = tmp_val - h  # 将当前元素减少h
        fxh2 = f(x)  # 计算 f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 使用中心差分公式计算梯度

        x[idx] = tmp_val  # 恢复当前元素的原始值
        it.iternext()  # 移动到下一个元素

    return grad
