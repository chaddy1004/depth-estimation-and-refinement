import tensorflow as tf

from tensorflow import keras
from keras.layers import Layer, Conv1D, Softmax

from utils.model_utils import Sine


class Classifier(Layer):
    def __init__(self, filter_channels, use_softmax=True):
        super(Classifier, self).__init__()
        self.filters = []
        self.filter_channels = filter_channels
        self.use_softmax = use_softmax
        self.conv1 = Conv1D(input_shape=self.nfeat, filters=self.filter_channels[1], padding="same")
        self.conv2 = Conv1D(filters=self.filter_channels[2], padding="same")
        self.conv3 = Conv1D(filters=self.filter_channels[3], padding="same")
        self.conv4 = Conv1D(filters=self.filter_channels[4], padding="same")
        self.sine = Sine()
        # sofrmax over vector that goes from 0 all the way to max disparity
        # therefore, argmax of the vector gives the disparity value
        self.softmax = Softmax

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.sine(x)
        x = self.conv1(tf.concat([x, inputs]))
        x = self.sine(x)
        x = self.conv2(tf.concat([x, inputs]))
        x = self.sine(x)
        x = self.conv3(tf.concat([x, inputs]))
        x = self.sine(x)
        x = self.conv4(tf.concat([x, inputs]))
        x = self.sine(x)
        if self.use_softmax:
            probs = self.softmax(x)
            return probs
        else:
            return x
