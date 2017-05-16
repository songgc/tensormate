import abc
import glob

import tensorflow as tf
from tensorflow.contrib import slim
from tensormate.graph.base import TfGgraphBuilder


class DataSetParams(object):

    def __init__(self, path_pattern_train, path_pattern_validation):
        self.path_pattern_train = path_pattern_train
        self.path_pattern_validation = path_pattern_validation
        self.samples_train = None
        self.samples_validation = None


class ImageDataSetParams(DataSetParams):

    def __init__(self, *args, **kwargs):
        super(ImageDataSetParams, self).__init__(*args, **kwargs)
        self.image_height = None
        self.image_weight = None
        self.image_channels = 3


class SupervisedLearningDataGenerator(TfGgraphBuilder):
    __metaclass__ = abc.ABCMeta

    def __init__(self, scope, dataset_params: DataSetParams,
                 is_training,
                 batch_size=64,
                 num_epochs=1,
                 num_threads=2,
                 shuffle_capacity=1000,
                 use_multi_reader=True,
                 prefetch_capacity=100,
                 prefetch_threads=2,
                 device="/cpu:0"):
        super(SupervisedLearningDataGenerator, self).__init__(scope=scope, device=device)
        self.dataset_params = dataset_params
        self.is_training = is_training
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.num_epochs = num_epochs
        self.shuffle_capacity = shuffle_capacity
        self.use_multi_readers = use_multi_reader
        self.prefetch_capacity = prefetch_capacity
        self.prefetch_threads = prefetch_threads

        dsparams = self.dataset_params
        if self.is_training:
            self.num_samples = dsparams.samples_train
            path_pattern = dsparams.path_pattern_train
        else:
            self.num_samples = dsparams.samples_validation
            path_pattern = dsparams.path_pattern_validation

        self.file_list = glob.glob(path_pattern)
        assert len(self.file_list) > 0

    @property
    def batch_num_per_epoch(self):
        return self.num_samples // self.batch_size

    @property
    def batch_num_limit(self):
        return self.num_epochs * self.num_samples // self.batch_size

    def _add_extra_ops(self, op_list, extra_op_name_list):
        if extra_op_name_list is not None:
            for name in extra_op_name_list:
                op_list.append(self._node_map.get(name))
        return op_list

    def _build(self, extra_op_name_list=None, debug=False):
        file_queue = tf.train.string_input_producer(self.file_list,
                                                    num_epochs=None,
                                                    shuffle=self.is_training,
                                                    capacity=max(self.num_threads, 32))
        if debug:
            data, label = self.parse_and_decode(file_queue)
            return self._add_extra_ops([data, label], extra_op_name_list)

        if self.is_training:
            if self.use_multi_readers:
                example_list = []
                for _ in range(self.num_threads):
                    with tf.name_scope("parse_and_decode"):
                        data, label = self.parse_and_decode(file_queue)
                    with tf.name_scope("preprocess_train"):
                        data, label = self.preprocess_train(data, label)
                    op_list = self._add_extra_ops([data, label], extra_op_name_list)
                    example_list.append(tuple(op_list))
                output_tensor_list = tf.train.shuffle_batch_join(
                    example_list,
                    batch_size=self.batch_size,
                    capacity=self.shuffle_capacity + 3 * self.batch_size,
                    min_after_dequeue=self.shuffle_capacity)
            else:
                with tf.name_scope("parse_and_decode"):
                    data, label = self.parse_and_decode(file_queue)
                with tf.name_scope("preprocess_train"):
                    data, label = self.preprocess_train(data, label)
                op_list = self._add_extra_ops([data, label], extra_op_name_list)
                output_tensor_list = tf.train.shuffle_batch(
                    op_list,
                    batch_size=self.batch_size,
                    num_threads=self.num_threads,
                    capacity=self.shuffle_capacity + 3 * self.batch_size,
                    min_after_dequeue=self.shuffle_capacity)
        else:
            with tf.name_scope("parse_and_decode"):
                data, label = self.parse_and_decode(file_queue)
            with tf.name_scope("preprocess_train"):
                data, label = self.preprocess_test(data, label)
            op_list = self._add_extra_ops([data, label], extra_op_name_list)
            output_tensor_list = tf.train.batch(
                op_list,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                capacity=1000 + 3 * self.batch_size
            )
        # another layer for prefetch
        if self.prefetch_capacity is not None and self.prefetch_capacity > 0:
            with tf.name_scope("batch_preprocess"):
                output_tensor_list = self.batch_preprocess(*output_tensor_list)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                output_tensor_list,
                capacity=self.prefetch_capacity,
                num_threads=self.prefetch_threads)
            output_tensor_list = batch_queue.dequeue()
        return output_tensor_list

    @abc.abstractmethod
    def parse_and_decode(self, file_queue):
        pass

    @abc.abstractmethod
    def preprocess_test(self, data, label):
        pass

    @abc.abstractmethod
    def preprocess_train(self, data, label):
        pass

    @abc.abstractmethod
    def batch_preprocess(self, *tensor_list):
        pass


ClassifierDataGenerator = SupervisedLearningDataGenerator
