import unittest
import numpy as np
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
    feature_sparse = features.SparseInt64Feature()


class FeaturesTest(unittest.TestCase):

    def setUp(self):
        self.value_a = 1
        self.value_b = -1.0
        self.value_c = b"Hello"
        self.value_sparse = [11, 12]

    def test_name(self):
        self.assertEqual(TestFeatures.feature_a.name, "feature/a")
        self.assertEqual(TestFeatures.feature_b.name, "feature_b")

    def test_example_encode_and_decode(self):
        # encode
        feature_tuples = [
            TestFeatures.feature_a(self.value_a),
            TestFeatures.feature_b(self.value_b),
            TestFeatures.feature_c(self.value_c),
            TestFeatures.feature_sparse(self.value_sparse)
        ]
        example = TestFeatures.to_pb_example(feature_tuples)
        # decode
        parsed = tf.parse_single_example(serialized=example.SerializeToString(), features=TestFeatures.feature_map())
        values = _run_tf_session(parsed)
        self.assertEqual(values[TestFeatures.feature_a.name], self.value_a)
        self.assertEqual(values[TestFeatures.feature_b.name], self.value_b)
        self.assertEqual(values[TestFeatures.feature_c.name], self.value_c)
        self.assertEqual(values[TestFeatures.feature_sparse.name].values.tolist(), self.value_sparse)

    def test_example_use_old_way(self):
        # encode
        feature_tuples = [
            ("feature/a", tf.train.Feature(int64_list=tf.train.Int64List(value=[self.value_a]))),
            ("feature_b", tf.train.Feature(float_list=tf.train.FloatList(value=[self.value_b]))),
            ("feature/c", tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.value_c]))),
            ("feature_sparse", tf.train.Feature(int64_list=tf.train.Int64List(value=self.value_sparse))),
        ]
        proto_features = tf.train.Features(feature=dict(feature_tuples))
        example = tf.train.Example(features=proto_features)
        # decode
        feature_map = {
            "feature/a": tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1),
            "feature_b": tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=-1),
            "feature/c": tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=""),
            "feature_sparse": tf.VarLenFeature(dtype=tf.int64),
        }
        parsed = tf.parse_single_example(serialized=example.SerializeToString(), features=feature_map)
        values = _run_tf_session(parsed)
        self.assertEqual(values["feature/a"], self.value_a)
        self.assertEqual(values["feature_b"], self.value_b)
        self.assertEqual(values["feature/c"], self.value_c)
        self.assertEqual(values["feature_sparse"].values.tolist(), self.value_sparse)


class TestSequenceFeatures(features.SequenceFeatures):
    length = features.Int64Feature()
    tokens = features.Int64FeatureList()
    labels = features.Int64FeatureList()
    bytes = features.BytesFeatureList()
    floats = features.Float32FeatureList()


class SequenceFeaturesTest(unittest.TestCase):

    def setUp(self):
        self.tokens = [1, 2, 3]
        self.labels = [0, 1, 0]
        self.floats = [0.1, 0.2]
        self.bytes = [b"ab", b"c"]

    def test_sequence_example(self):
        # encode
        feature_tuples = [
            TestSequenceFeatures.length(len(self.tokens)),
            TestSequenceFeatures.tokens(self.tokens),
            TestSequenceFeatures.labels(self.labels),
            TestSequenceFeatures.floats(self.floats),
            TestSequenceFeatures.bytes(self.bytes),
        ]
        seq_ex = TestSequenceFeatures.to_pb_sequence_example(feature_tuples=feature_tuples)
        # decode
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=seq_ex.SerializeToString(),
            context_features=TestSequenceFeatures.context_feature_map(),
            sequence_features=TestSequenceFeatures.feature_list_map()
        )
        context, sequence = _run_tf_session([context_parsed, sequence_parsed])
        self.assertEqual(context[TestSequenceFeatures.length.name], len(self.tokens))
        self.assertEqual(sequence[TestSequenceFeatures.tokens.name].tolist(), self.tokens)
        self.assertEqual(sequence[TestSequenceFeatures.labels.name].tolist(), self.labels)
        self.assertEqual(sequence[TestSequenceFeatures.bytes.name].tolist(), self.bytes)
        np.testing.assert_almost_equal(sequence[TestSequenceFeatures.floats.name].tolist(), self.floats,
                                       decimal=7, verbose=True)

    def test_sequence_example_use_old_way(self):
        # encode
        seq_ex = tf.train.SequenceExample()
        sequence_length = len(self.tokens)
        seq_ex.context.feature["length"].int64_list.value.append(sequence_length)
        fl_tokens = seq_ex.feature_lists.feature_list["tokens"]
        fl_labels = seq_ex.feature_lists.feature_list["labels"]
        fl_floats = seq_ex.feature_lists.feature_list["floats"]
        fl_bytes = seq_ex.feature_lists.feature_list["bytes"]
        for token, label in zip(self.tokens, self.labels):
            fl_tokens.feature.add().int64_list.value.append(token)
            fl_labels.feature.add().int64_list.value.append(label)
        for flt, byte in zip(self.floats, self.bytes):
            fl_floats.feature.add().float_list.value.append(flt)
            fl_bytes.feature.add().bytes_list.value.append(byte)
        # decode
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "floats": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "bytes": tf.FixedLenSequenceFeature([], dtype=tf.string),
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=seq_ex.SerializeToString(),
            context_features=context_features,
            sequence_features=sequence_features
        )
        context, sequence = _run_tf_session([context_parsed, sequence_parsed])
        self.assertEqual(context["length"], len(self.tokens))
        self.assertEqual(sequence["tokens"].tolist(), self.tokens)
        self.assertEqual(sequence["labels"].tolist(), self.labels)
        self.assertEqual(sequence[TestSequenceFeatures.bytes.name].tolist(), self.bytes)
        np.testing.assert_almost_equal(sequence[TestSequenceFeatures.floats.name].tolist(), self.floats,
                                       decimal=7, verbose=True)


if __name__ == '__main__':
    unittest.main()
