import numpy as np

# 1. 仅提供 stop (隐式 start=0, step=1)
# 对应重载: def arange(stop: _IntLike_co...) -> _Array1D[signedinteger]
arr1 = np.arange(5)
print(arr1)
# 结果: [0 1 2 3 4] (整数类型)

# 2. 提供 start, stop, step
# 对应重载: def arange(start: _IntLike_co, stop: _IntLike_co, step: _IntLike_co...)
arr2 = np.arange(1, 10, 2)
print(arr2)
# 结果: [1 3 5 7 9]

# 3. 浮点数步长
# 对应重载: def arange(start: _FloatLike_co, stop: _FloatLike_co, step: _FloatLike_co...) -> _Array1D[floating]
arr3 = np.arange(0, 1, 0.2)
print(arr3)
# 结果: [0.  0.2 0.4 0.6 0.8] (浮点数类型)

# 4. 显式指定 dtype
arr4 = np.arange(0, 5, dtype=np.float32)
print(arr4)
# 结果: [0. 1. 2. 3. 4.] (强制转换为 32 位浮点数)

# 与 Python 的 range() 函数的对比👇
# range() 只可处理整数 (Integer)
for i in range(0, 5, 1):
    print(i)  # 输出 0, 1, 2, 3, 4

a1 = range(1, 10, 2).index(5)  # 2
a2 = range(1, 10, 2).count(4)  # 0
a3 = range(1, 10, 2).stop  # 10
print(a1)
print(a2)
print(a3)

# NOTE 数组元组自乘
arr5 = np.arange(0, 4)
print(f"arr5 = ", arr5)
print(f"arr5**2 = ", arr5**2)
arr6 = np.arange(0, 4).reshape(2, 2)
print(f"arr6 = ", arr6)
print(f"arr6**2 = ", arr6**2)
