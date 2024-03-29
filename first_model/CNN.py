import numpy as np
from keras.datasets import mnist
from keras.models import load_model
import keras
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255
y_test_cat = keras.utils.to_categorical(y_test, 10)


model = load_model('cnn_model.keras')

# loss, accuracy = model.evaluate(x_test, y_test_cat)

