# imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
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
#
# # Get image classification
# classes = ['airplanes', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# print("Image class is", classes[y_train[idx][0]])

# Convert labels to a set of 10 numbers for inputs
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# # Print new labels
# print(y_train_one_hot)
#
# # Print new label of current picture
# print("The one hot label is", y_train_one_hot[idx])

# Normalize pixels to be values in the range [0, 1]
x_train = x_train / 255
x_test = x_test / 255
# print(x_train[idx])

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

# Add a layer with 1000 neurons (connected to previous layers)
model.add(Dense(1000, activation='relu'))

# Add a dropout layer
model.add(Dropout(0.5))

# Add a layer with 500 neurons (connected to previous layers)
model.add(Dense(1000, activation='relu'))

# Add a another dropout layer
model.add(Dropout(0.5))

# Add a layer with 250 neurons
model.add(Dense(250, activation='relu'))

# Add a layer with 10 neurons b/c we have 10 different classifications
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
hist = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.2)

# # Save the model?
# model.save("my_model")
#
# # # Load the model?
# # model = keras.models.load_model("my_model")



# evaluate the model using the test data set
print(model.evaluate(x_test, y_test_one_hot)[1])

# Visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc="upper left")
plt.show()

# Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc="upper right")
plt.show()
