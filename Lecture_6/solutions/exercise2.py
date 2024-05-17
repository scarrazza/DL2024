#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load data
X_train = np.load('training_data.npy')
y_train = np.load('training_label.npy')

X_test = np.load('test_data.npy')
y_test = np.load('test_label.npy')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(30, activation='relu', input_shape=(X_train.shape[1], 1)))
# or
# model.add(tf.keras.layers.LSTM(30, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
# model.add(tf.keras.layers.LSTM(30, activation='relu'))
# or
#model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30, activation='relu'), input_shape=(X_train.shape[1], 1)))
model.add(tf.keras.layers.Dense(1))

# training
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=25, batch_size=32)
y_predicted = model.predict(X_test)


# Show results
plt.figure(figsize=(10,8))

plt.subplot(3, 1, 1)
plt.plot(y_train, label='Train data')
plt.plot(range(len(y_train), len(y_train)+len(y_test)), y_test, 'k', label='Test data')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("All data")

plt.subplot(3, 2, 3)
plt.plot(y_test, color='k', label = 'True value')
plt.plot(y_predicted, color='red', label = 'Predicted')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("Predicted data (full test set)")

plt.subplot(3, 2, 4)
plt.plot(y_test[0:100], color='k', label = 'True value')
plt.plot(y_predicted[0:100], color = 'red', label='Predicted')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("Predicted data (first 100 days)")

plt.subplot(3, 2, 5)
plt.plot(y_test-y_predicted, color='k')
plt.ylabel("Residual")
plt.xlabel("Day")
plt.title("Residual plot")

plt.subplot(3, 2, 6)
plt.scatter(y_predicted, y_test, s=2, color='black')
plt.ylabel("Y true")
plt.xlabel("Y predicted")
plt.title("Scatter plot")

mse = np.mean(np.square(y_test - y_predicted))
print(f"MSE = {mse}")

plt.subplots_adjust(hspace = 0.5, wspace=0.3)
plt.show()
