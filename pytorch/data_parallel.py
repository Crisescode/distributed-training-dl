import argparse
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import pyramidnet
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Data Parallel Training')

parser.add_argument('--train-dir', '-td', type=str, default="./train_dir",
                    help='the path that the model saved (default: "./train_dir")')
parser.add_argument('--batch-size', '-b', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-workers', type=int, default=4, help='')
parser.add_argument('--test-batchsize', '-tb', type=int, default=1000,
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

args = parser.parse_args()


def main():
    # set run env
    if args.gpu_nums > 1:
        device = 'cuda' if torch.cuda.is_available() else "cpu"
        gpu_ids = ','.join([str(id) for id in range(args.gpu_nums)])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    else:
        raise ValueError("gpu-nums must be greater than 1.")

    print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root='/home/zhaopp5', train=True, download=True,
                            transform=transforms_train)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    print('==> Making model..')

    model = pyramidnet()
    if args.gpu_nums > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        train(epoch, model, criterion, optimizer, train_loader, device)

    if args.save_model:
        torch.save(model.state_dict(), "./data_parallel_model.pt")


def train(epoch, model, criterion, optimizer, train_loader, device):
    model.train()

    train_loss, correct, total = 0, 0, 0
    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
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
            print('Epoch[{}]: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                epoch, batch_idx, len(train_loader), train_loss / (batch_idx + 1), acc, batch_time))

    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))


if __name__ == '__main__':
    main()
