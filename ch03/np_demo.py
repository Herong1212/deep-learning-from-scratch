import numpy as np

A = np.array([1, 2, 3, 4])
print(A)

# 数组的维数可以通过 np.dim() 函数获得
print(np.ndim(A))  # 一维数组

# 数组的形状可以通过实例变量 shape 获得
# !这里的 A.shape 的结果是个元组（tuple）。这是因为一维数组的情况下也要返回和多维数组的情况下一致的结果。
# 例如，二维数组时返回的是元组 (4,3)，三维数组时返回的是元组 (4,3,2)，因此一维数组时也同样以元组的形式返回结果
print(A.shape)
print(A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)
