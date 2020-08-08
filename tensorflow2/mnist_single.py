#!/usr/bin/python
# -*-coding:utf-8 -*-

import os
import argparse

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers


# create cnn model
class Net(object):
    def __init__(self):
        model = models.Sequential()
        model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model


# inital dateset
class DataSet(object):
    def __init__(self):
        data_path = os.path.dirname(os.path.realpath(__file__)) \
                    + '/../../datasets/mnist/mnist.npz'
        (train_images, train_labels), (test_images, test_labels) = \
            datasets.mnist.load_data(path=data_path)

        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))

        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels


class PrintLR(tf.keras.callbacks.Callback):
    def __init__(self, lr):
        super(PrintLR, self).__init__()
        self.lr = lr

    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, self.lr))


# train and val
class Train:
    def __init__(self):
        self.model = Net().model
        self.data = DataSet()

    def train(self, args):
        # Define the checkpoint directory to store the checkpoints
        checkpoint_dir = args.train_dir
        # Name of the checkpoint files
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt_{epoch}")


        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=args.train_dir, histogram_freq=1),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                               save_weights_only=True),
        ]

        self.model.compile(optimizer=optimizers.Adam(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.fit(self.data.train_images, self.data.train_labels,
                       batch_size=args.batch_size,
                       epochs=args.epochs,
                       callbacks=callbacks,
                       validation_data=(self.data.test_images, self.data.test_labels))

        # EVAL
        self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        eval_loss, eval_acc = self.model.evaluate(
            self.data.test_images, self.data.test_labels, verbose=2)
        print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))


def main():
    # training params settings
    parser = argparse.ArgumentParser(description='Tensorflow 2.0 MNIST Example')
    parser.add_argument('--train_dir', '-td', type=str, default='./train_dir',
                        help='the folder of svaing model')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batchsize', '-tb', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', '-sm', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()

    app = Train()
    app.train(args)


if __name__ == "__main__":
   main()
