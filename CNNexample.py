# Another example found here:
# I will be using this as a baseline to learn
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras import datasets, layers, models
from keras.utils import np_utils
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization
import matplotlib as plt
import numpy as np
from os import walk
from playsound import playsound

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = train_images / 255
test_images = test_images / 255

num_classes = 10
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

# Building the model... that's a lot of layers...
model = Sequential()

model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

# Output layer
model.add(layers.Dense(num_classes, activation='softmax'))    # num_classes = 10

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

hist = model.fit(train_images, train_labels, batch_size=64, epochs=100, validation_data=(test_images, test_labels))

print("TESTING...")
res = model.evaluate(test_images, test_labels)[1]

filenames = next(walk("images\\"), (None, None, []))[2]
IMAGES = []
for name in filenames:
    path = "images\\" + name
    IMAGES.append(np.array([plt.imread(path)]))

guesses = []
for k in range(len(IMAGES)):
    prediction = model.predict(IMAGES[k])
    idx = []
    for p in range(10):
        idx.append(p)

    for i in range(10):
        for j in range(10):
            if prediction[0][idx[j]] > prediction[0][idx[k]]:
                temp = idx[i]
                idx[i] = idx[j]
                idx[j] = temp
    guesses.append(class_names[idx[0]])

model.save("example_model")

print("Test accuracy is", res)
for i in range(len(guesses)):
    print(class_names[i], "prediction is", guesses[i], "-", class_names[i] == guesses[i])

# Don't mind this part. Feel free to comment out... or not
playsound("sounds\\alarm.mp3")
