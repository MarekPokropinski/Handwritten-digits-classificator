import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.datasets.mnist import load_data

import os
import numpy as np
import matplotlib.pyplot as plt


def build_model():
    model = Sequential([
        Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        GlobalMaxPooling2D(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def plot(x, y):
    fig, ax = plt.subplots(5, 5)
    for i in range(25):
        ax[i % 5, i//5].imshow(x[i], cmap='gray')
        ax[i % 5, i//5].set_title(y[i])
    plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')
    plot(x_train[:25], y_train[:25])
    (x_train, y_train), (x_test, y_test) = (np.expand_dims(x_train, -1),
                                            np.expand_dims(y_train, -1)),    (np.expand_dims(x_test, -1), np.expand_dims(y_test, -1))
    datagen = ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True)
    datagen.fit(x_train)

    # print(datagen.mean, datagen.std)
    # print(x_train[10], y_train[0])
    plot(x_train[:25], y_train[:25])

    model = build_model()

    # model.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=datagen.flow(x_test, y_test, batch_size=32),
    #           steps_per_epoch=len(x_train) / 32, epochs=2)

    # model.save('model.h5')
