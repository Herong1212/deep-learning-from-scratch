import numpy as np

dict1 = {"Product": "Apple", "Price": 100}
dict1["num"] = 2
print(dict1)
dict1.keys()
print(f"dict1.keys(): ", dict1.keys())

tel = {"jack": 4098, "sape": 4139}
tel["guido"] = 4127
print(f"tel: ", tel)

print(tel["jack"])

del tel["sape"]
print(f"tel: ", tel)

print(list(tel.keys()))
print(list(tel))

tinydict = {"Name": "Runoob", "Age": 7, "Class": "First"}
for k, v in tinydict.items():
    print(k, v)

for i, v in enumerate(tinydict):
    print(i, v)

for k in tinydict.keys():
    print(k)
for v in tinydict.values():
    print(v)

print("###################################################")
# 1. 定义 X 轴和 Y 轴上的一维刻度
x = np.array([1, 2, 3])
y = np.array([10, 20])

print("原始 x:", x)
print("原始 y:", y)

# 2. 召唤 meshgrid 生成网格矩阵
X, Y = np.meshgrid(x, y)

print("\n--- 经过 meshgrid 转换后 ---")
print("矩阵 X:\n", X)
print("矩阵 Y:\n", Y)
