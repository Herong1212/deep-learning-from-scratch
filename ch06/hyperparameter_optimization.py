# coding: utf-8
import sys, os

# sys.path.append(os.pardir)  # 将父目录添加到系统路径中，以便导入父目录中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

# 加载 MNIST 数据集，并进行归一化处理
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 为了加快训练速度，减少训练数据量
x_train = x_train[:500]  # shappe=(500, 784)
t_train = t_train[:500]  # shape=(500,)

# 打乱训练数据
x_train, t_train = shuffle_dataset(x_train, t_train)

# 设置验证数据的比例，以分割验证数据
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)  # validation_num = 100

# 分割出验证数据和训练数据
x_val = x_train[:validation_num]  # shape=(100, 784)
t_val = t_train[:validation_num]  # shape=(100,)
x_train = x_train[validation_num:]  # shape=(400, 784)
t_train = t_train[validation_num:]  # shape=(400,)


def __train(lr, weight_decay, epocs=50):
    """
    训练多层神经网络模型并返回准确率列表

    参数:
        lr (float): 学习率
        weight_decay (float): 权重衰减系数
        epocs (int): 训练轮数，默认为 50

    返回:
        tuple: 包含测试准确率列表和训练准确率列表的元组
    """

    # 创建多层神经网络模型
    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100, 100],  # 极其深的网络（6个隐藏层）
        output_size=10,
        weight_decay_lambda=weight_decay,  # 传入权值衰减系数！
    )

    # 创建训练器对象
    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_val,
        t_val,
        epochs=epocs,
        mini_batch_size=100,
        optimizer="sgd",
        optimizer_param={"lr": lr},
        verbose=False,
    )

    # 开始训练
    trainer.train()

    # 返回测试准确率列表和训练准确率列表
    return trainer.test_acc_list, trainer.train_acc_list


# 进行超参数的随机搜索 ======================================
optimization_trial = 100
results_val = {}
results_train = {}

# 执行超参数随机搜索
for _ in range(optimization_trial):
    # 指定要探索的超参数范围 ===============
    weight_decay = 10 ** np.random.uniform(-8, -4) # 从 -8 到 -4 之间随机抽取一个小数（比如 -5.3）
    lr = 10 ** np.random.uniform(-6, -2)
    # ================================================

    # 训练模型并获取准确率列表
    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print(
        "val acc:"
        + str(val_acc_list[-1])
        + " | lr:"
        + str(lr)
        + ", weight decay:"
        + str(weight_decay)
    )

    # 将结果存储在字典中
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 绘制图表========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

# 绘制最佳超参数组合的准确率曲线
for key, val_acc_list in sorted(
    results_val.items(), key=lambda x: x[1][-1], reverse=True
):
    print("Best-" + str(i + 1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i + 1)
    plt.title("Best-" + str(i + 1))
    plt.ylim(0.0, 1.0)
    if i % 5:
        plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

# plt.show()
plt.savefig("ch06/hyperparameter_optimization.png")
