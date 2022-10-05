import tensorflow as tf




mnist = tf.keras.datasets.mnist





(x_train, y_train),(x_test, y_test) = mnist.load_data()





import matplotlib.pyplot as plt




x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_text = tf.keras.utils.normalize(x_test, axis = 1)


import numpy as np
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE,1)
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE,1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, Activation, MaxPooling2D

model = Sequential()



model.add(Conv2D(64, (3,3),input_shape = x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))


model.add(Dense(10))
model.add(Activation('softmax'))


model.compile(loss = "sparse_categorical_crossentropy", optimizer= "adam", metrics=['accuracy'])

model.fit(x_trainr, y_train,epochs=5, validation_split= 0.3)
model.save('models/MyModel.h5')