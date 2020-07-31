# SingleNode and Distributed Implementation with Pytorch 

## 介绍
这个目录主要是`Pytorch`框架的单机以及分布式训练示例。示例是使用`PyramidNet`模型在`cifar10`数据集上进行验证

* `model.py`：这个文件主要用以实现模型封装。
* `single_gpu.py`：这个文件是单机单卡训练脚本。
* `data_parallel.py`：这个文件是使用`data`进行模型并行的单机多卡训练脚本。
* `distibuted_data_parallel.py`：这个文件是实现分布式训练，可通过给定不同参数实现单机多卡、多机单卡以及多机多卡分布式训练。

## Requirements
* python3.+
* torch==1.5.0
* torchvision==0.6.0

## 训练

### 单机单卡
关于单机单卡，就是使用单个结点单个gpu进行训练，这应该也是大家最常用的训练方式。

* 执行命令：
```
python single_gpu.py --gpu-nums 1 --epochs 2 --batch-size 64 --train-dir /home/crise/single_gpu --dataset-dir /home/crise/cifar10 --log-interval 20  --save-model
```
上面命令也可简便执行如下：
```
python single_gpu.py -g 1 -e 2 -b 64 -td /home/crise/single_gpu -dd /home/crise -li 20  -sm
```
* 参数介绍：
    * --gpu-nums: 使用gpu的数量，其实只能等于1（因为是单卡训练），不然会报`ValueError`，默认值为0。
    * --epochs: 最大`epoch`数量，默认值为3。
    * --batch-size: batch size 大小，默认值为64。
    * --train-dir: 模型参数及结果存放路径，默认值为`./train_dir`。
    * --dataset-dir: 数据集存放路径，默认值为`./data`。
    * --log-interval: 日志打印频率，默认值为迭代20步打印一次。
    * --save-model: 是否需要存储模型，带上这个参数则存，否则不存。

* 训练时间：
  本来是想贴图片的，但发现贴上来很难看。可以点击[训练时长](../imgs/pytorch/sg_time.PNG)以及[GPU利用率](../imgs/pytorch/sg_gpu.PNG)查看。
  * batch time: 0.255s
  * epoch time: 03:20min
  * gpu util: 98%

### 单机多卡
单机多卡有两种实现方式，一种是使用`DataParallel`接口实现数据并行单机多卡分布式训练，另外一个是使用`DistributedDataParallel`接口实现

#### `DataParallel`实现
这个方式主要是通过单个进程，关于该接口实现详细介绍请参考博客[分布式训练之PyTorch]

* 执行命令：
```
python data_parallel.py --gpu-nums 2 --epochs 2 --batch-size 64 --train-dir /home/crise/data_parallel --dataset-dir /home/crise/cifar10 --log-interval 20  --save-model
```

  > 注：简化执行命令可参照单机单卡训练。

* 参数介绍：参照单机单卡。

* 训练时间：
  [训练时长](../imgs/pytorch/data_parallel_time.PNG) 与 [训练时间](../imgs/pytorch/data_parallel_gpu.PNG)，图中能看出在同一个进程中使用了两个gpu进行训练。
  * batch time: 0.170s
  * epoch time: 02:18min
  * gpu util: 80%

#### `DistributedDataParallel` 实现
这个方式会通过`torch.multiprocessing`来启动多个进程，进行

* 执行命令：

  * Shell 1:
    ```
    CUDA_VISIBLE_DEVICES='0' python distributed_data_parallel.py --epochs 2 --batch-size 64 --train-dir /home/crise/single_node_distribute --dataset-dir /home/crise/cifar10 --log-interval 20  --save-model --world-size 2 --rank 0
    ```
  * 同一个节点 Shell 2 执行:
    ```
    CUDA_VISIBLE_DEVICES='1' python distributed_data_parallel.py --epochs 2 --batch-size 64 --train-dir /home/crise/single_node_distribute --dataset-dir /home/crise/cifar10 --log-interval 20  --save-model --world-size 2 --rank 1
    ```

  > 注：简化执行命令可参照单机单卡训练。

* 参数介绍：基本参数参考单机单卡，新增参数如下。
  * --world-size: 启动的进程总数，默认值为1。
  * --rank: 当前进程序号，默认值为0。

* 训练时间：[训练时长](../imgs/pytorch/single_node_distribute_rank0_time.PNG) 与 [训练时间](../imgs/pytorch/single_node_distribute.PNG)，图中能看出是有两个进程中分别在不同的gpu上进行训练。
  * batch time: 0.274s
  * epoch time: 01:51min
  * gpu0 util: 98%
  * gpu1 util: 99%
  
### 多机多卡分布式
多机多卡分布式训练还是主要通过`DistributedDataParallel`接口来实现，
* 执行命令:

  * Node 1 & Shell 1 执行：
    ```
    CUDA_VISIBLE_DEVICES='0' python distributed_data_parallel.py --epochs 2 --batch-size 64 --train-dir /home/crise/multi_node_distribute --dataset-dir /home/crise --log-interval 20  --save-model --init-method tcp://c1:20201 --world-size 4 --rank 0
    ```
  * Node 1 & Shell 2 执行：
    ```
    CUDA_VISIBLE_DEVICES='1' python distributed_data_parallel.py --epochs 2 --batch-size 64 --train-dir /home/crise/multi_node_distribute --dataset-dir /home/crise --log-interval 20  --save-model --init-method tcp://c1:20201 --world-size 4 --rank 1
    ```

  * Node 2 & Shell 1 执行：
    ```
    CUDA_VISIBLE_DEVICES='0' python distributed_data_parallel.py --epochs 2 --batch-size 64 --train-dir /home/crise/multi_node_distribute --dataset-dir /home/crise --log-interval 20  --save-model --init-method tcp://c1:20201 --world-size 4 --rank 2
    ```

  * Node 2 & Shell 2 执行：
    ```
     CUDA_VISIBLE_DEVICES='0' python distributed_data_parallel.py --epochs 2 --batch-size 64 --train-dir /home/crise/multi_node_distribute --dataset-dir /home/crise --log-interval 20  --save-model --init-method tcp://c1:20201 --world-size 4 --rank 3
    ```
  > 注：简化执行命令可参照单机单卡训练。

* 参数介绍：基本参数参考单机单卡，新增参数如下。
  * --world-size: 启动的进程总数，默认值为1。
  * --rank: 当前进程序号，默认值为0。
  * --init-method：初始化方式

* 训练时间：[训练时长](../imgs/pytorch/multi_node_distribute_time.PNG) 与 [训练时间](../imgs/pytorch/multi_node_distribute_gpu.PNG)，这只是一个节点GPU截图，另一个节点大差不差。
  * batch time: 0.301s
  * epoch time: 01:07min
  * gpu0 util: 99%
  * gpu1 util: 99%

## 性能对比 
上面几种训练都是在`Tesla P100`，显存为`16Gb`，batch_size 为64 


