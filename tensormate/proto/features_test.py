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
        # encode
        feature_tuples = [
            TestFeatures.feature_a(value_a),
            TestFeatures.feature_b(value_b),
            TestFeatures.feature_c(value_c),
        ]
        example = TestFeatures.to_pb_example(feature_tuples)
        # decode
        parsed = tf.parse_single_example(serialized=example.SerializeToString(), features=TestFeatures.feature_map())
        values = _run_tf_session(parsed)
        self.assertEqual(values[TestFeatures.feature_a.name], value_a)
        self.assertEqual(values[TestFeatures.feature_b.name], value_b)
        self.assertEqual(values[TestFeatures.feature_c.name], value_c)

    def test_example_use_old_way(self):
        value_a = 1
        value_b = -1.0
        value_c = b"Hello"
        # encode
        feature_tuples = [
            ("feature/a", tf.train.Feature(int64_list=tf.train.Int64List(value=[value_a]))),
            ("feature_b", tf.train.Feature(float_list=tf.train.FloatList(value=[value_b]))),
            ("feature/c", tf.train.Feature(bytes_list=tf.train.BytesList(value=[value_c]))),
        ]
        features = tf.train.Features(feature=dict(feature_tuples))
        example = tf.train.Example(features=features)
        # decode
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


class TestSequenceFeatures(features.SequenceFeatures):
    length = features.Int64Feature()
    tokens = features.Int64FeatureList()
    labels = features.Int64FeatureList()


class SequenceFeaturesTest(unittest.TestCase):

    def test_sequence_example(self):
        tokens, labels = [1, 2, 3], [0, 1, 0]
        # encode
        feature_tuples = [
            TestSequenceFeatures.length(len(tokens)),
        ]
        feature_list_tuples = [
            TestSequenceFeatures.tokens(tokens),
            TestSequenceFeatures.labels(labels),
        ]
        seq_ex = TestSequenceFeatures.to_pb_sequence_example(feature_tuples=feature_tuples,
                                                             feature_list_tuples=feature_list_tuples)
        # decode
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=seq_ex.SerializeToString(),
            context_features=TestSequenceFeatures.context_feature_map(),
            sequence_features=TestSequenceFeatures.feature_list_map()
        )
        context, sequence = _run_tf_session([context_parsed, sequence_parsed])
        self.assertEqual(context[TestSequenceFeatures.length.name], len(tokens))
        self.assertEqual(sequence[TestSequenceFeatures.tokens.name].tolist(), tokens)
        self.assertEqual(sequence[TestSequenceFeatures.labels.name].tolist(), labels)

    def test_sequence_example_use_old_way(self):
        tokens, labels = [1, 2, 3], [0, 1, 0]
        # encode
        seq_ex = tf.train.SequenceExample()
        sequence_length = len(tokens)
        seq_ex.context.feature["length"].int64_list.value.append(sequence_length)
        fl_tokens = seq_ex.feature_lists.feature_list["tokens"]
        fl_labels = seq_ex.feature_lists.feature_list["labels"]
        for token, label in zip(tokens, labels):
            fl_tokens.feature.add().int64_list.value.append(token)
            fl_labels.feature.add().int64_list.value.append(label)
        # decode
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=seq_ex.SerializeToString(),
            context_features=context_features,
            sequence_features=sequence_features
        )
        context, sequence = _run_tf_session([context_parsed, sequence_parsed])
        self.assertEqual(context["length"], len(tokens))
        self.assertEqual(sequence["tokens"].tolist(), tokens)
        self.assertEqual(sequence["labels"].tolist(), labels)


if __name__ == '__main__':
    unittest.main()
