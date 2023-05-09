import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, Layer

class Sine(Layer):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, inputs):
        return tf.math.sin(self.w0 * inputs)

