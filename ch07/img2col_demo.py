import sys, os

# sys.path.append(os.pardir)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def im2col_verbose(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    带打印输出的 im2col 观察器
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    print(f"1. 准备就绪：输入原图形状 {(N, C, H, W)}，计划滑动 {out_h * out_w} 次。")
    print(f"   最终大矩阵预期形状: ({N * out_h * out_w}, {C * filter_h * filter_w})\n")

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 执行填充
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 为了方便人类观察，我们在做最后的 reshape 前，手动模拟一下提取过程
    print("-" * 40)
    print("🎬 开始慢动作回放滑动窗口提取过程：")
    step = 0
    # 我们按照 N -> out_h -> out_w 的顺序，模拟放大镜每次停顿的位置
    for n in range(N):
        for oh in range(out_h):
            for ow in range(out_w):
                step += 1
                # 提取出当前窗口框住的所有数据
                window_data = img[
                    n,
                    :,
                    oh * stride : oh * stride + filter_h,
                    ow * stride : ow * stride + filter_w,
                ]
                # 把框住的数据拉平
                flattened = window_data.flatten()
                print(f" [第 {step} 步] 放大镜停在 (y={oh*stride}, x={ow*stride})")
                print(f"          框住的矩阵:\n{window_data[0]}")  # 打印第一个通道看看
                print(f"          被拉平后变成新的一行 -> {flattened}\n")

    print("-" * 40)
    # 框架底层的真实暴力折叠操作
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


# ==========================================
# 实验 1：用极简的 1~16 顺序数字来观察！
# ==========================================
print("========== 实验 1：极简透视观察 ==========")
# 生成一个 1 张图，1 个通道，4x4 大小的矩阵，填入 1 到 16
x_simple = np.arange(1, 17).reshape(1, 1, 4, 4)
print("【最原始的图片】:")
print(x_simple[0, 0])
print(x_simple[0][0])
print("【最原始的图片形状】:")
print(x_simple.shape)
print("\n")

# 使用 3x3 的滤波器去扫它
col_simple = im2col_verbose(x_simple, filter_h=3, filter_w=3, stride=1, pad=0)

print("🏆 【im2col 最终生成的巨大的二维矩阵 col】:")
print(col_simple)
print(f"最终形状: {col_simple.shape}\n")


# ==========================================
# 实验 2：解密您原代码中的形状
# ==========================================
print("========== 实验 2：您的原代码形状验证 ==========")
from common.util import im2col  # 调回书本的函数

# 1. 单张图片
x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(f"x1 (1张图) 经过 im2col 后形状: {col1.shape}")

# 2. 批处理：10张图片
x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(f"x2 (10张图) 经过 im2col 后形状: {col2.shape}")
