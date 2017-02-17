import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope

from tensormate.graph import TfGgraphBuilder


class ImageGraphBuilder(TfGgraphBuilder):

    def __init__(self, scope=None, device=None, data_format="NHWC",
                 data_format_ops=(layers.conv2d,
                                  layers.convolution2d,
                                  layers.convolution2d_transpose,
                                  layers.max_pool2d,
                                  layers.batch_norm)):
        super(ImageGraphBuilder, self).__init__(scope=scope, device=device)
        self.data_format = data_format
        self.data_format_ops = data_format_ops if data_format_ops is not None else []

    def __call__(self, *args, **kwargs):
        is_training = kwargs.get("is_training", True)
        reuse = self.ref_count > 0 and not is_training
        g = tf.get_default_graph().as_graph_def()
        existing_nodes = set([node.name for node in g.node])
        with tf.variable_scope(self.scope, reuse=reuse):
            with tf.device(self._device), \
                 arg_scope(self.data_format_ops, data_format=self.data_format):
                output = self._build(*args, **kwargs)
            self._call_count += 1
        if self._call_count == 1:
            self._trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope)
        g = tf.get_default_graph().as_graph_def()
        self._created_nodes = [node for node in g.node if node.name not in existing_nodes]
        return output
