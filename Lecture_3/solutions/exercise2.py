#!/usr/bin/env python
import numpy as np
import tensorflow as tf

# fix random seed
tf.random.set_seed(0)

n_input = 1
n_hidden_1 = 5
n_hidden_2 = 2
n_output = 1

weights = {
    'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random.normal([n_hidden_2, n_output]))
}

biases = {
    'b1': tf.Variable(tf.random.normal([n_hidden_1])),
    'b2': tf.Variable(tf.random.normal([n_hidden_2])),
    'out': tf.Variable(tf.random.normal([n_output]))
}

# Create model
def multilayer_perceptron(x):
    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# print predictions
x = np.linspace(-2, 2, 10, dtype=np.float32).reshape(-1, 1)
y1 = multilayer_perceptron(x)

# Sequential model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(n_hidden_1, activation='sigmoid', input_dim=1))
model.add(tf.keras.layers.Dense(n_hidden_2, activation='sigmoid'))
model.add(tf.keras.layers.Dense(n_output, activation='linear'))
model.summary()

# assign parameters from previous model
model.set_weights([weights["h1"], biases["b1"],
                   weights["h2"], biases["b2"],
                   weights["out"], biases["out"]])
y2 = model.predict(x)

if not np.allclose(y1, y2):
    raise ValueError("results do not match")
