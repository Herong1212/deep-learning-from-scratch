深度学习入门：基于Python的理论与实现 (Deep Learning from Scratch)
==========================


[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/images/deep-learning-from-scratch.png" width="200px">](https://www.oreilly.co.jp/books/9784873117584/)

本书是《Deep Learning from Scratch》（O'Reilly Japan 出版，中文译名通常为《深度学习入门：基于 Python 的理论与实现》）的配套网站。本仓库包含了书中所使用的全部源代码。


## 目录结构

|文件夹名    |说明                         |
|:--        |:--                          |
|ch01       |第 1 章使用的源代码            |
|ch02       |第 2 章使用的源代码            |
|...        |...                          |
|ch08       |第 8 章使用的源代码            |
|common     |各章节共用的函数及类           |
|dataset    |数据集相关的处理代码           |


注意： 关于源代码的详细讲解，请参阅原书内容。

## Python 与依赖库
运行本项目代码需要安装以下软件及库：

* Python 3.x
* NumPy
* Matplotlib

※ 请务必使用 Python 3 系列版本。

## 运行方法

进入各章节对应的文件夹，运行相应的 Python 命令。

```
$ cd ch01
$ python man.py

$ cd ../ch05
$ python train_nueralnet.py
```

## 在云端服务运行

您可以直接点击下表中的按钮，在 AWS 提供的免费计算环境 [Amazon SageMaker Studio Lab](https://studiolab.sagemaker.aws/) 上运行本书的代码（注：需事先通过邮箱地址[进行注册](https://studiolab.sagemaker.aws/requestAccount)）。
关于 SageMaker Studio Lab 的具体使用方法，请[参阅](https://github.com/aws-sagemaker-jp/awesome-studio-lab-jp/blob/main/README_usage.md)此处说明。您也可以在 [Amazon SageMaker Studio Lab Community](https://github.com/aws-studiolab-jp/awesome-studio-lab-jp) 获取最新资讯。

|文件夹名称 |Amazon SageMaker Studio Lab
|:--        |:--                          |
|ch01       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch01.ipynb)|
|ch02       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch02.ipynb)|
|ch03       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch03.ipynb)|
|ch04       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch04.ipynb)|
|ch05       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch05.ipynb)|
|ch06       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch06.ipynb)|
|ch07       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch07.ipynb)|
|ch08       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/ch08.ipynb)|
|common       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch/blob/master/notebooks/common.ipynb)|


## 开源协议

本仓库中的源代码采用[MIT 许可证](http://www.opensource.org/licenses/MIT)。
无论是商业还是非商业用途，均可自由使用。

## 勘误表

本书的勘误信息发布在以下页面：

https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata

如果您发现了页面中未列出的排版错误或内容疏漏，请通过邮件告知：[japan@oreilly.co.jp](<mailto:japan@oreilly.co.jp>)。

