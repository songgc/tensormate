import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope

from tensormate.graph import TfGgraphBuilder


class ImageGraphBuilder(TfGgraphBuilder):

    def __init__(self, scope=None, device=None, plain=False, data_format="NHWC",
                 data_format_ops=(layers.conv2d,
                                  layers.convolution2d,
                                  layers.convolution2d_transpose,
                                  layers.convolution2d_in_plane,
                                  layers.convolution2d_transpose,
                                  layers.conv2d_in_plane,
                                  layers.conv2d_transpose,
                                  layers.separable_conv2d,
                                  layers.separable_convolution2d,
                                  layers.avg_pool2d,
                                  layers.max_pool2d,
                                  layers.batch_norm)):
        super(ImageGraphBuilder, self).__init__(scope=scope, device=device, plain=plain)
        self.data_format = data_format
        self.data_format_ops = data_format_ops if data_format_ops is not None else []

    def _call_body(self, *args, **kwargs):
        # is_training = kwargs.get("is_training", True)
        # reuse = self.ref_count > 0
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            with arg_scope(self.data_format_ops, data_format=self.data_format):
                if self._device is None:
                    output = self._build(*args, **kwargs)
                else:
                    with tf.device(self._device):
                        output = self._build(*args, **kwargs)
        return output
