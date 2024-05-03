#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class_names = ['T-shirt/top', 'Trouser', 'Pullover',
               'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle boot'] # categories 10 objects


def plot_sample(images, labels):
    """Plot utils."""
    plt.figure(figsize=(10,10))
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


def plot_predictions(predictions, images, labels):
    rows = 5
    cols = 3
    plt.figure(figsize=(4*cols, 2*rows))
    for i in range(rows * cols):
        plt.subplot(rows, 2*cols, 2 * i + 1)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.yticks([])
        plt.xticks([])
        predicted_label = np.argmax(predictions[i])
        if predicted_label == labels[i]:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions[i]):2.0f}% ({class_names[labels[i]]})", color=color)

        plt.subplot(rows, 2*cols, 2 * i + 2)
        tp = plt.bar(range(10), predictions[i], color='grey')
        tp[predicted_label].set_color('red')
        tp[labels[i]].set_color('blue')
        plt.yticks([])
        plt.xticks([])
        plt.ylim([0,1])
    plt.show()


def create_model():
    """"Model definition."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    # output layer -> 10 units / categories
    # softmax: np.sum([out[i] for i in range(10)]) == 1
    return model


def main(epochs):
    # load dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    print('training shape:', train_images.shape, train_labels.shape)
    print('test shape:', test_images.shape, test_labels.shape)

    # normalize data
    norm = np.max(train_images)
    print('Max values:', norm)
    train_images = train_images / norm
    test_images = test_images / norm

    plot_sample(train_images, train_labels)

    # build model
    model = create_model()
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test set loss:", test_loss)
    print("Test set accuracy:", test_acc)

    predictions = model.predict(test_images)
    plot_predictions(predictions, test_images, test_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line options.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    args = parser.parse_args()
    main(args.epochs)
