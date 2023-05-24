import tensorflow as tf
from keras.applications import VGG16, ResNet50
from keras.layers import Layer

BACKBONES = {"VGG16": VGG16(include_top=False), "ResNet50": ResNet50(include_top=False)}


def get_backbone(config):
    try:
        return BACKBONES[config.exp.backbone]
    except KeyError:
        raise KeyError("Proper backbone needed")


class Sine(Layer):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, inputs):
        return tf.math.sin(self.w0 * inputs)
