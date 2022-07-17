# imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from os import walk
# Load data
from keras.datasets import cifar10


plt.style.use("fivethirtyeight")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# # Look at variable data types
# print(type(x_train))
# print(type(y_train))
# print(type(x_test))
# print(type(y_test))
# print()
# # Get the shapes
# print("x_train shape:", x_train.shape)
# print("y_train shape:", y_train.shape)
# print("x_test shape:", x_test.shape)
# print("y_test shape:", y_test.shape)
#
# # Look at the first image as an array
# idx = 1
# print(x_train[idx])
#
# # # Show the first image as a picture
# # img = plt.imshow(x_train[idx])
#
# # Get the image label
# print("Image label:", y_train[idx])

# Get image classification
classes = ['airplanes', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# print("Image class is", classes[y_train[idx][0]])

# Convert labels to a set of 10 numbers for inputs
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# # Print new labels
# print(y_train_one_hot)
#
# # Print new label of current picture
# print("The one hot label is", y_train_one_hot[idx])
#
# Normalize pixels to be values in the range [0, 1]
x_train = x_train / 255
x_test = x_test / 255

# Create the models architecture
model = Sequential()

# Add first Layer
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))

# Add pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add a second convolution layer
model.add(Conv2D(32, (5, 5), activation='relu'))

# Add a second pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add a flattening layer (converts to linear array)
model.add(Flatten())

# Add a layer with 2048 neurons (connected to previous layers)
model.add(Dense(2048, activation='relu'))

# Add a dropout layer
model.add(Dropout(0.5))

# Add a layer with 1024 neurons (connected to previous layers)
model.add(Dense(1024, activation='relu'))

# Add a another dropout layer
model.add(Dropout(0.5))

# Add a layer with 512 neurons
model.add(Dense(512, activation='relu'))

# Add a layer with 128 neurons
model.add(Dense(128, activation='relu'))

# Add a layer with 32 neurons
model.add(Dense(32, activation='relu'))

# Add a layer with 10 neurons b/c we have 10 different classifications
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', metrics=['accuracy'])

# Train the model
hist = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.2)

# # Save the model?
# model.save("my_model")
#
# # Load the model?
# model = keras.models.load_model("my_model")
#
# # Retrain model after loading
# model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.2)
#
# for i in range(25, 0, -1):
#     print("Trial", (26 - i))
#     model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=i/100)
#     print("Accuracy is", model.evaluate(x_test, y_test_one_hot)[1], '\n')
#
# # Save again
# model.save("my_model")

# evaluate the model using the test data set
test_accuracy = model.evaluate(x_test, y_test_one_hot)[1]

# # Visualize the models accuracy
# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.title("Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Val"], loc="upper left")
# plt.show()
#
# # Visualize the models loss
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title("Model Loss")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Val"], loc="upper right")
# plt.show()

# Test on sample data
filenames = next(walk("images\\"), (None, None, []))[2]
IMAGES = []
for name in filenames:
    path = "images\\" + name
    IMAGES.append(np.array([plt.imread(path)]))

model = keras.models.load_model("my_model")
guesses = []
for i in range(len(IMAGES)):
    prediction = model.predict(IMAGES[i])
    idx = []
    for p in range(10):
        idx.append(p)

    for j in range(10):
        for k in range(10):
            if prediction[0][idx[j]] > prediction[0][idx[k]]:
                temp = idx[j]
                idx[j] = idx[k]
                idx[k] = temp
    guesses.append(classes[idx[0]])

print("Test accuracy is", test_accuracy)
for i in range(len(guesses)):
    print(classes[i], "prediction is", guesses[i], "-", classes[i] == guesses[i])
