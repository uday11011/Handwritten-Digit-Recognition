from ctypes.wintypes import PINT
from turtle import back
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import colorama
from colorama import Back, Fore, Style
colorama.init()
print(Fore.BLUE)
model = load_model('models/MyModel.h5')
img = cv2.imread('C:/Users/UDAY/OneDrive/Desktop/handwritten project/pic9.jpeg')

IMG_SIZE = 28

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)




newimg1 = tf.keras.utils.normalize(resized, axis = 1)

newimg1= np.array(newimg1).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



predictionsed1 = model.predict(newimg1)

import colorama
from colorama import Back, Fore, Style
colorama.init()

print(Back.YELLOW)
print(Style.BRIGHT)
print(Fore.MAGENTA+"THE DIGIT IS PREDICTED AS:")

print( np.argmax(predictionsed1))
print()
print(Style.RESET_ALL)
