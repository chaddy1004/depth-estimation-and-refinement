import tensorflow as tf

from tensorflow import keras
from keras.layers import Layer, Conv1D, Softmax

from utils.model_utils import Sine, get_backbone

from models.classifier import Classifier
from models.regressor import Regressor


class Refiner(Layer):
    def __init__(self, config, filter_channels, use_softmax=True):
        super(Refiner, self).__init__()
        self.config = config
        self.backbone = get_backbone(config=config)
        self.max_disp = self.config.exp.max_disp
        self.height = self.config.data.input_h
        self.width = self.config.data.input_w

        self.backbone_output_size = self.backbone.layers[-1].output_shape[-1]  # channel last
        self.classification = Classifier(filter_channels=[self.backbone_output_size, 512, 256, 128, self.max_disp])
        self.regressor = Regressor(self.backbone_output_size + 1, 128, 64, 1)

    def query(self, points, labels):
        if labels is not None:
            self.labels = labels
        u = scale_coords(points[:, 0:1, :], self.width)
        v = scale_coords(points[:, 1:2, :], self.height)
        uv = torch.cat([u, v], 1)
        self.uv = uv

    def call(self, img, disp, points, labels):
        x = disp

        x = self.backbone(x)
        self.query()
