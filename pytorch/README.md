# SingleNode and Distributed Implementation with Pytorch 

## Introduction
这个目录主要是`Pytorch`框架的单机以及分布式训练示例。示例是使用`PyramidNet`模型在`cifar10`数据集上进行验证

* `model.py`：这个文件主要用以实现模型封装。
* `single_gpu.py`：这个文件是单机单卡训练脚本。
* `data_parallel.py`：这个文件是使用`data`进行模型并行的单机多卡训练脚本。
* `distibuted_data_parallel.py`：这个文件是实现分布式训练，可通过给定不同参数实现单机多卡、多机单卡以及多机多卡分布式训练。

## Requirements
* python3.+
* torch==1.5.0
* torchvision==0.6.0

## Training

### 单机单卡训练
关于单机单卡，就是使用单个结点单个gpu进行训练，这应该也是大家最常用的训练方式。

* 执行命令：
```
python single_gpu.py --gpu-nums 1 --epochs 2 --batch-size 64 --train-dir /home/crise/single_gpu --dataset-dir /home/crise --log-interval 20  --save-model

```
上面命令也可简便执行如下：
```
python single_gpu.py -g 1 -e 2 -b 64 -td /home/crise/single_gpu -dd /home/crise -li 20  -sm

```
* 参数介绍：
    * --gpu-nums：使用gpu的数量，其实只能等于1（因为是单卡训练），不然会报`ValueError`，默认值为0。
    * --epochs：最大`epoch`数量，默认值为3。
    

### 单机多卡训练
单机多卡有两种实现方式，一种是使用`DataParallel`接口实现数据并行单机多卡分布式训练，另外一个是使用`DistributedDataParallel`接口实现

#### `DataParallel`实现
这个方式

* 执行命令：
```
python single_gpu.py --gpu-nums 1 --epochs 2 --batch-size 64 --train-dir /home/crise/single_gpu --dataset-dir /home/crise --log-interval 20  --save-model

```
同样简便执行命令可参照单机单卡训练执行。

* 参数介绍：


#### `DistributedDataParallel` 实现
这个方式

* 执行命令：
```
python single_gpu.py --gpu-nums 1 --epochs 2 --batch-size 64 --train-dir /home/crise/single_gpu --dataset-dir /home/crise --log-interval 20  --save-model

```
同样简便执行命令可参照单机单卡训练执行。

* 参数介绍：


#### 多机多卡分布式训练

