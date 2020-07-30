## SingleNode and Distributed Implementation with Pytorch 

### Introduction
这个目录主要是`Pytorch`框架的单机以及分布式训练示例。示例是使用`PyramidNet`模型在`cifar10`数据集上进行验证

* `model.py`：这个文件主要用以实现模型封装。
* `single_gpu.py`：这个文件是单机单卡训练脚本。
* `data_parallel.py`：这个文件是使用`data`进行模型并行的单机多卡训练脚本。
* `distibuted_data_parallel.py`：这个文件是实现分布式训练，可通过改变参数实现多机单卡以及多机多卡训练。

