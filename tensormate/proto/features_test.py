import unittest
import tensorflow as tf
from tensormate.proto import features


def _run_tf_session(fetches):
    with tf.train.SingularMonitoredSession() as sess:
        out = sess.run(fetches)
    return out


class TestFeatures(features.Features):
    feature_a = features.Int64Feature(replace={"_": "/"})
    feature_b = features.Float32Feature()
    feature_c = features.BytesFeature(replace={"_": "/"})


class FeaturesTest(unittest.TestCase):

    def test_name(self):
        self.assertEqual(TestFeatures.feature_a.name, "feature/a")
        self.assertEqual(TestFeatures.feature_b.name, "feature_b")

    def test_example_encode_and_decode(self):
        value_a = 1
        value_b = -1.0
        value_c = b"Hello"
        feature_tuples = [
            TestFeatures.feature_a(value_a),
            TestFeatures.feature_b(value_b),
            TestFeatures.feature_c(value_c),
        ]
        example = TestFeatures.to_pb_example(feature_tuples)
        parsed = tf.parse_single_example(serialized=example.SerializeToString(), features=TestFeatures.feature_map())
        values = _run_tf_session(parsed)
        self.assertEqual(values[TestFeatures.feature_a.name], value_a)
        self.assertEqual(values[TestFeatures.feature_b.name], value_b)
        self.assertEqual(values[TestFeatures.feature_c.name], value_c)

    def test_example_use_old_way(self):
        value_a = 1
        value_b = -1.0
        value_c = b"Hello"
        feature_tuples = [
            ("feature/a", tf.train.Feature(int64_list=tf.train.Int64List(value=[value_a]))),
            ("feature_b", tf.train.Feature(float_list=tf.train.FloatList(value=[value_b]))),
            ("feature/c", tf.train.Feature(bytes_list=tf.train.BytesList(value=[value_c]))),
        ]
        features = tf.train.Features(feature=dict(feature_tuples))
        example = tf.train.Example(features=features)
        feature_map = {
            "feature/a": tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1),
            "feature_b": tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=-1),
            "feature/c": tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=""),
        }
        parsed = tf.parse_single_example(serialized=example.SerializeToString(), features=feature_map)
        values = _run_tf_session(parsed)
        self.assertEqual(values["feature/a"], value_a)
        self.assertEqual(values["feature_b"], value_b)
        self.assertEqual(values["feature/c"], value_c)


if __name__ == '__main__':
    unittest.main()
