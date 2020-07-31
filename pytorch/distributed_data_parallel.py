import argparse
import datetime
import time

from os import mkdir, path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from model import pyramidnet
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Distributed Training')

parser.add_argument('--train-dir', '-td', type=str, default="./train_dir",
                    help='the path that the model saved (default: "./train_dir")')
parser.add_argument('--dataset-dir', '-dd', type=str, default="./data",
                    help='the path of dataset (default: "./data")')
parser.add_argument('--batch-size', '-b', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-workers', type=int, default=4, help='')
parser.add_argument('--test-batch-size', '-tb', type=int, default=1000,
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', '-e', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--gpu-nums', '-g', type=int, default=0,
                    help='Number of GPU in each mini-batch')
parser.add_argument('--learning-rate', '--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', '-sm', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4, metavar='W',
                    help='weight decay(default: 1e-4)')
parser.add_argument('--init-method', default='tcp://127.0.0.1:13456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world-size', default=1, type=int, help='')

args = parser.parse_args()


def main():
    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node: ", ngpus_per_node)
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    # init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                            world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(gpu)

    print("From Rank: {}, Use GPU: {} for training".format(args.rank, gpu))

    print('From Rank: {}, ==> Making model..'.format(args.rank))
    net = pyramidnet()
    net.cuda(gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    print("batch_size: ", args.batch_size)

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu], output_device=gpu)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('From Rank: {}, The number of parameters of model is'.format(args.rank), num_params)

    print('From Rank: {}, ==> Preparing data..'.format(args.rank))
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root=args.dataset_dir, train=True, download=True,
                            transform=transforms_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=args.num_workers,
                              sampler=train_sampler)

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        train(epoch, net, criterion, optimizer, train_loader, args.rank)
        scheduler.step()

    if args.save_model:
        if not path.exists(args.train_dir):
            mkdir(args.train_dir)

        # if args.rank == 0:
        torch.save(
            net.module.state_dict(),
            path.join(
                args.train_dir,
                "distributed_data_parallel_{}.pth".format(args.rank)
            )
        )
        print("From Rank: {}, model saved.".format(args.rank))


def train(epoch, net, criterion, optimizer, train_loader, rank):
    net.train()

    train_loss, correct, total = 0, 0, 0
    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()

        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100 * correct / total

        batch_time = time.time() - start

        if batch_idx % args.log_interval == 0:
            print('From Rank: {}, Epoch:[{}][{}/{}]| loss: {:.3f} | '
                  'acc: {:.3f} | batch time: {:.3f}s '.format(
                   rank, epoch, batch_idx, len(train_loader),
                   train_loss / (batch_idx + 1), acc, batch_time), flush=True)

    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("From Rank: {}, Training time {}".format(rank, elapse_time))


if __name__ == '__main__':
    main()
