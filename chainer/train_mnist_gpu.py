#!/usr/bin/env python
import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
import mnist_dataset

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    # This script is almost identical to train_mnist.py. The only difference is
    # that this script uses data-parallel computation on two GPUs.
    # See train_mnist.py for more details.
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=400,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use gpu')
    parser.add_argument('--gpu_number', '-n', type=int, default=1,
                        help='Number of gpus')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    if args.gpu:
       print("Use the gpu environment to perform work, please \
set the number of gpus you need to use.")
       if args.gpu_number == 2: 
          print('===================================')    
          use_gpu = {'main': 0, 'second': 1}
          print('# use GPU:2')
          print('# unit: {}'.format(args.unit))
          print('# minibatch-size: {}'.format(args.batchsize))
          print('# epoch: {}'.format(args.epoch))
          print('===================================') 
       elif args.gpu_number == 1:
          print('===================================')
          use_gpu = {'main': 0}
          print('# use GPU: 1')
          print('# unit: {}'.format(args.unit))
          print('# minibatch-size: {}'.format(args.batchsize))
          print('# epoch: {}'.format(args.epoch))
          print('===================================')
       else:
          raise ValueError('please set the correct number of gpus you need to use!')
    else:
       raise ValueError('gpu env set error!') 
	   
    chainer.backends.cuda.get_device_from_id(0).use()

    model = L.Classifier(MLP(args.unit, 10))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = mnist_dataset.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # ParallelUpdater implements the data-parallel gradient computation on
    # multiple GPUs. It accepts "devices" argument that specifies which GPU to
    # use.
    updater = training.updaters.ParallelUpdater(
        train_iter,
        optimizer,
        # The device of the name 'main' is used as a "master", while others are
        # used as slaves. Names other than 'main' are arbitrary.
        devices=use_gpu,
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    # trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
