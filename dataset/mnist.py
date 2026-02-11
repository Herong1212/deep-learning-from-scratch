# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError("You should use Python 3.x")
import os.path
import gzip
import pickle
import os
import numpy as np


# url_base = 'http://yann.lecun.com/exdb/mnist/'
url_base = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # mirror site

key_file = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}

# __file__：这是 Python 的内置变量（类似 C++ 的宏 __FILE__）。它代表当前这个脚本文件（即 dataset/mnist.py）的路径。
# os.path.abspath(...)：把上面的路径变成绝对路径。即：/home/user/fishbook/deep-learning/dataset/mnist.py
# os.path.dirname(...)：去掉文件名，只保留目录。即：/home/user/fishbook/deep-learning/dataset/
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"
    }
    request = urllib.request.Request(url_base + file_name, headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(file_path, mode="wb") as f:
        f.write(response)
    print("Done")


def download_mnist():
    for v in key_file.values():
        _download(v)


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def _convert_numpy():
    dataset = {}
    dataset["train_img"] = _load_img(key_file["train_img"])
    dataset["train_label"] = _load_label(key_file["train_label"])
    dataset["test_img"] = _load_img(key_file["test_img"])
    dataset["test_label"] = _load_label(key_file["test_label"])

    return dataset


def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """加载MNIST数据集

    Parameters
    ----------
    normalize : bool, 可选。是否将图像像素值归一化到 0.0~1.0。默认为 True。
                如果将该参数设置为 False, 则输入图像的像素会保持原来的 0~255。
    one_hot_label : bool, 可选。是否以 one-hot 数组的形式返回标签。默认为 False。
        one-hot数组是指例如 [0,0,1,0,0,0,0,0,0,0] 这样的数组。
        当 one_hot_label 为 False 时，只是像 7、2 这样简单保存正确解标签；
        当 one_hot_label 为 True 时，标签则保存为 one-hot 表示。
    flatten : bool, 可选。是否将图像展平为一维数组。默认为 True。。
                如果将该参数设置为 False，则输入图像为 1 × 28 × 28 的三维数组；
                若设置为 True，则输入图像会保存为由 784 个元素构成的一维数组。

    返回
    -------
    tuple
        以(训练图像, 训练标签), (测试图像, 测试标签) 的形式返回数据。
        每个元素都是numpy数组。
    """
    # 如果MNIST数据集文件不存在，则进行初始化处理
    if not os.path.exists(save_file):
        init_mnist()

    # 加载保存的数据集文件
    with open(save_file, "rb") as f:
        dataset = pickle.load(f)

    # 如果normalize为True，则将图像数据的像素值归一化到 0.0~1.0
    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # 如果 one_hot_label 为 True，则将标签转换为 one-hot 数组
    if one_hot_label:
        dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_one_hot_label(dataset["test_label"])

    # 如果 flatten 为 False，则将图像数据 reshape 为 28 x 28 的二维数组
    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    # 以元组形式返回训练数据和测试数据
    return (dataset["train_img"], dataset["train_label"]), (
        dataset["test_img"],
        dataset["test_label"],
    )


if __name__ == "__main__":
    init_mnist()
