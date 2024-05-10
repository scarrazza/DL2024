#!/usr/bin/env python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import matplotlib.pyplot as plt
import numpy as np


def plot_sample(images, labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])
    plt.show()


def train(features, labels, parameters):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(units=parameters['layer_size'], activation='relu'))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(lr=parameters['learning_rate'])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    model.fit(features, labels, epochs=5)
    return model


def test(model, features, labels):
    acc = model.evaluate(features, labels)
    return acc


def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # feature scaling
    train_images = train_images / 255
    test_images = test_images / 255

    print('train:', train_images.shape)
    print('test:', test_images.shape)

    plot_sample(train_images, train_labels)

    # build model
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    test_acc = test(model, test_images, test_labels)
    print(f'Test accuracy {test_acc}')

    # implement hyperparameter tune with hyperopt
    test_setup = {
        'layer_size': 2,
        'learning_rate': 1.0
    }
    model = train(train_images, train_labels, test_setup)
    print('Accuracy on test set is:', test(model, test_images, test_labels)[1])

    # objective function
    def hyper_func(params):

        model = train(train_images, train_labels, params)
        test_acc = test(model, test_images, test_labels)

        return {'loss': -test_acc[1], 'status': STATUS_OK}

    search_space = {
        'layer_size': hp.choice('layer_size', np.arange(10, 100, 20)),
        'learning_rate': hp.loguniform('learning_rate', -10, 0)
    }

    trials = Trials()
    best = fmin(hyper_func, search_space, algo=tpe.suggest, max_evals=5, trials=trials)
    print(space_eval(search_space, best))

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    xs = [t['tid'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]
    ax1.set_xlim(xs[0]-1, xs[-1]+1)
    ax1.scatter(xs, ys, s=20)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')

    xs = [t['misc']['vals']['layer_size'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]

    ax2.scatter(xs, ys, s=20)
    ax2.set_xlabel('Layers')
    ax2.set_ylabel('Accuracy')

    xs = [t['misc']['vals']['learning_rate'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]

    ax3.scatter(xs, ys, s=20)
    ax3.set_xlabel('learning_rate')
    ax3.set_ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
