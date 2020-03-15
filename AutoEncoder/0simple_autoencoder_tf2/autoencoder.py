# coding: utf-8

# https://towardsdatascience.com/machine-learning-autoencoders-712337a07c71

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt # for plots

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.round(x_train, 0)
x_test = np.round(x_test, 0)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)

# inputs = tf.keras.layers.Input(784)
inputs = tf.keras.layers.Input((784,))
encoded_1 = tf.keras.layers.Dense(128)(inputs)
encoded = tf.keras.layers.Dense(64, activation='relu')(encoded_1)
decoded_1 = tf.keras.layers.Dense(128)(encoded)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(decoded_1)

auto_encoder = tf.keras.Model(inputs, decoded)
auto_encoder.compile(loss='binary_crossentropy', 
                     optimizer='adam', 
                     metrics=['accuracy'])
auto_encoder.summary()
# tf.keras.utils.plot_model(auto_encoder, show_shapes=True, to_file='autoenc.png', dpi=200)

# 参数重用很简单
encoder = tf.keras.Model(inputs, encoded)
encoder.summary()
# tf.keras.utils.plot_model(encoder, show_shapes=True, to_file='enc.png', dpi=200)



auto_encoder.fit(x_train, x_train, 
                 epochs=10,
                 batch_size=256,
                 shuffle=True)

predicted_2dim = encoder.predict(x_test)
predicted_original = auto_encoder.predict(x_test)

shown = {}
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=1, wspace=0.1)
i = 0
for data, y in zip(predicted_original, y_test):
    if y not in shown and y==len(shown):
        i += 1
        ax = fig.add_subplot(1, 10, i)
        ax.text(1, -1, str(y), fontsize=25, ha='center', c='g')
        ax.imshow(np.array(data).reshape(28, 28), cmap='gray')
    shown[y] = True
    if len(shown) == 10:
        break
