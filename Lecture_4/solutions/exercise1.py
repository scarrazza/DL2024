#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def create_baseline_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), loss="mean_squared_error")
    return model


def create_nn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, input_dim=1, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), loss="mean_squared_error")
    return model


def plot_data(X, Y, color, title):
    plt.figure()
    plt.scatter(X, Y, color=color)
    plt.grid()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")


def plot_history(history, title):
    plt.figure()
    plt.plot(history.epoch, np.array(history.history["loss"]), label="Train loss")
    plt.plot(history.epoch, np.array(history.history["val_loss"]), label="Val loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Mean squared error)")
    plt.legend()


def plot_results(X, Y, Y_predict, title):
    plt.figure()
    plt.scatter(X, Y, color="blue")
    plt.plot(X, Y_predict, color="red")
    plt.grid()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")


data = np.loadtxt('../data.dat')
X_train = data[:,0]
Y_train = data[:,1]
X_val = data[:,2]
Y_val = data[:,3]

plot_data(X_train, Y_train, "blue", "Training data")
plot_data(X_val, Y_val, "red", "Validation data")

model = create_baseline_model()
history = model.fit(X_train, Y_train, batch_size=X_train.shape[0], epochs=500, validation_data=(X_val, Y_val))
plot_history(history, "baseline")

model2 = create_nn_model()
history = model2.fit(X_train, Y_train, batch_size=X_train.shape[0], epochs=500, validation_data=(X_val, Y_val))
plot_history(history, "Neural Net")

plot_results(X_val, Y_val, model.predict(X_val), "Baseline")
plot_results(X_val, Y_val, model2.predict(X_val), "Neural Net")

plt.show()
