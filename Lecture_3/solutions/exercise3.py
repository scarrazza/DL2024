#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.random.set_seed(0)


# Truth data
def f(x):
    return x * 3.0 + 2.0


# Generates artificial data
def generate_data():
    # A vector of random x values
    x = tf.linspace(-2, 2, 200)
    x = tf.cast(x, tf.float32)

    # Generate some noise
    noise = tf.random.normal(shape=x.shape)

    # Calculate y
    y = f(x) + noise
    return x, y


# My custom model
class MyModel(tf.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.w * x + self.b


# My custom model
class MyKerasModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x, training=False):
        return self.w * x + self.b


# MSE loss function
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


# Given a callable model, inputs, outputs, and a learning rate...
def train(model, x, y, learning_rate):

    with tf.GradientTape() as t:
        current_loss = loss(y, model(x))

    # Use GradientTape to calculate the gradients with respect to W and b
    dw, db = t.gradient(current_loss, [model.w, model.b])

    # Subtract the gradient scaled by the learning rate
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


# Define a training loop
def report(model, loss):
    return f"W = {model.w.numpy():1.2f}, b = {model.b.numpy():1.2f}, loss={loss:2.5f}"


def training_loop(model, x, y, epochs):

    # Collect the history of W-values and b-values to plot later
    weights = []
    biases = []

    for epoch in epochs:
        # Update the model with the single giant batch
        train(model, x, y, learning_rate=0.1)

        # Track this before I update
        weights.append(model.w.numpy())
        biases.append(model.b.numpy())
        current_loss = loss(y, model(x))

        print(f"Epoch {epoch:2d}:")
        print("    ", report(model, current_loss))

    return weights, biases


def main():
    # Data generation
    x, y = generate_data()

    # Custom model allocation
    model = MyModel()
    ypred = model(x)

    # Training loop
    current_loss = loss(y, model(x))
    print("Untrained model loss: %1.6f" % loss(model(x), y).numpy())

    print(f"Starting:")
    epochs = range(10)
    print("    ", report(model, current_loss))
    weights, biases = training_loop(model, x, y, epochs)
    print("Trained loss: %1.6f" % loss(model(x), y).numpy())

    # Plot results
    plt.figure()
    plt.plot(epochs, weights, 'r', label='weights')
    plt.plot(epochs, [3.0] * len(epochs), 'r--', label = "True weight")
    plt.plot(epochs, biases, 'b', label='bias')
    plt.plot(epochs, [2.0] * len(epochs), "b--", label="True bias")
    plt.legend()

    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x, f(x), "orange", label="Ground truth")
    plt.plot(x, ypred, "green", label="Untrained predictions")
    plt.plot(x, model(x), "red", label="Trained predictions")
    plt.title("Functional API")
    plt.legend()
    plt.show()

    # keras model
    keras_model = MyKerasModel()
    #weights, biases = training_loop(keras_model, x, y, epochs)
    keras_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                        loss=tf.keras.losses.mean_squared_error)
    keras_model.fit(x, y, epochs=10, batch_size=len(x))

    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x, f(x), "orange", label="Ground truth")
    plt.plot(x, ypred, "green", label="Untrained predictions")
    plt.plot(x, keras_model(x), "red", label="Trained predictions")
    plt.title("Keras Model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
