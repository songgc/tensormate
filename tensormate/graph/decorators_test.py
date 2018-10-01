import tensorflow as tf
from tensorflow.contrib import layers
from tensormate.graph import shape_info, graph_info, name_scope
from pprint import pprint

tf.logging.set_verbosity(tf.logging.INFO)


class ShapeInfoTest(tf.test.TestCase):

    cached = True

    @shape_info(cached=cached)
    def fwd_graph(self, inputs, num_outputs):
        return layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[3, 3])

    def test(self):
        input_shape = [10, 24, 24, 3]
        inputs = tf.Variable(tf.zeros(shape=input_shape))
        outputs = self.fwd_graph(inputs, num_outputs=16)
        if ShapeInfoTest.cached:
            pprint(self.fwd_graph.result)
            input_name = "Variable:0"
            output_name = "Conv/Relu:0"
            self.assertAllEqual(self.fwd_graph.result[0][-1], input_shape)
            output_expected_shape = input_shape.copy()
            output_expected_shape[-1] = 16
            self.assertAllEqual(self.fwd_graph.result[1][-1], output_expected_shape)


class NameScopeTest(tf.test.TestCase):

    # @graph_info(cached=True)
    @shape_info(cached=True)
    @name_scope("my_scope")
    def fwd_graph(self, inputs, num_outputs):
        return layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[3, 3])

    def test(self):
        input_shape = [10, 24, 24, 3]
        inputs = tf.Variable(tf.zeros(shape=input_shape))
        outputs = self.fwd_graph(inputs, num_outputs=16)
        print(self.fwd_graph.result)


class GraphInfoTest(tf.test.TestCase):

    cached = True

    @graph_info(cached=cached)
    def fwd_graph(self, inputs, num_outputs, scope="test", reuse=False):
        return layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[3, 3], scope=scope, reuse=reuse)

    def test(self):
        input_shape = [10, 24, 24, 3]
        inputs = tf.Variable(tf.zeros(shape=input_shape))
        inputs = tf.identity(inputs)
        outputs = self.fwd_graph(inputs, num_outputs=16)
        if GraphInfoTest.cached:
            self.check_result("/home/guocong/git/github/tensormate/test1.html")
        outputs_1 = self.fwd_graph(inputs, num_outputs=16, reuse=True)
        if GraphInfoTest.cached:
            self.check_result("/home/guocong/git/github/tensormate/test2.html")
        self.assertEqual(self.fwd_graph.count, 2)
        self.assertEqual(self.fwd_graph.__name__, self.fwd_graph.__wrapped__.__name__)

    def check_result(self, output_file):
        pprint(self.fwd_graph.result)
        html = self.fwd_graph.viz_html_string
        output_file = output_file
        with open(output_file, "tw") as f:
            f.write(html)


if __name__ == '__main__':
    tf.test.main()
