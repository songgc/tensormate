import unittest
import os
from tensormate import graph
import tensorflow as tf
from tensorflow.contrib import layers
from pprint import pprint


TMP_DIR = "/tmp/tensormate"


class SubGraph(graph.TfGgraphBuilder):

    def _build(self, inputs):
        x = layers.conv2d(inputs, 16, 3)
        x = layers.conv2d(x, 64, 3, stride=2)
        return x


class BaseTest(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR)

    def tearDown(self):
        print("end")

    def test_subgraph(self):
        print("subgraph")
        inputs = tf.placeholder(tf.float32, [32, 224, 224, 3])
        g1 = SubGraph(scope="subgraph1")
        g2 = SubGraph(scope="subgraph2")
        y = g1(inputs)
        y = g2(y)
        g1.visualize(os.path.join(TMP_DIR, "v1.html"))
        g2.visualize(os.path.join(TMP_DIR, "v2.html"))
        pprint(g2.op_counting())


if __name__ == '__main__':
    unittest.main()
