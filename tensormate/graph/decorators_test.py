import tensorflow as tf
from tensorflow.contrib import layers
from tensormate.graph import shape_info, graph_info
from pprint import pprint

tf.logging.set_verbosity(tf.logging.INFO)


class ShapeInfoTest(tf.test.TestCase):

    cached = True

    @shape_info(cached=cached)
    def graph(self, inputs, num_outputs):
        return layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[3, 3])

    def test(self):
        input_shape = [10, 24, 24, 3]
        inputs = tf.Variable(tf.zeros(shape=input_shape))
        outputs = self.graph(inputs, num_outputs=16)
        if ShapeInfoTest.cached:
            pprint(self.graph.result)
            input_name = "Variable:0"
            output_name = "Conv/Relu:0"
            self.assertAllEqual(self.graph.result[0][-1], input_shape)
            output_expected_shape = input_shape.copy()
            output_expected_shape[-1] = 16
            self.assertAllEqual(self.graph.result[1][-1], output_expected_shape)


class GraphInfoTest(tf.test.TestCase):

    cached = True

    @graph_info(cached=cached)
    def graph(self, inputs, num_outputs):
        return layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[3, 3])

    def test(self):
        input_shape = [10, 24, 24, 3]
        inputs = tf.Variable(tf.zeros(shape=input_shape))
        inputs = tf.identity(inputs)
        outputs = self.graph(inputs, num_outputs=16)
        if GraphInfoTest.cached:
            pprint(self.graph.result)
            html = self.graph.viz_html_string
            output_file = "/home/guocong/git/github/tensormate/test1.html"
            with open(output_file, "tw") as f:
                f.write(html)


if __name__ == '__main__':
    tf.test.main()
