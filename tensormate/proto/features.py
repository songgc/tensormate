from collections import OrderedDict

import six
import tensorflow as tf


class Feature(object):
    """Base class for Feature
    
    Args:
      name: feature name.
      dtype: data type of input.
      shape: data shape of input.
      default: default value for serialization if value is None or value to be used if an example is missing 
      in deserialization.
      replace: a dictionary in which a string key is replaced by the corresponding value.   
    """
    def __init__(self, name, dtype, shape=[], default=None, replace=None):
        self._name = name
        self.dtype = dtype
        self.shape = shape
        self.default = default
        self.replace = replace

    @staticmethod
    def int64_feature(value):
        """Static method for converting int64 features into Feature proto.
        
        Args:
          value: Value to be serialized. 
        Returns:
          Serialized Feature proto with int64 type.
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
        """Static method for converting float features into Feature proto.
        
        Args:
          value: Value to be serialized. 
        Returns:
          Serialized Feature proto with float type.
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def bytes_feature(value):
        """Static method for converting bytes features into Feature proto.
        
        Args:
          value: Value to be serialized. 
        Returns:
          Serialized Feature proto with bytes type.
        """
        if isinstance(value, str):
            value = str.encode(value)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _encode(value):
        """Abstract encode method."""
        raise NotImplementedError

    def __call__(self, value=None):
        """Output a tuple of feature name and serialized Feature proto.
        
        Returns:
          a tuple of feature name and serialized Feature proto.
        """
        if value is None:
            value = self.default
        return self.name, self._encode(value)

    @property
    # TODO change pars_type to parsing_type
    def parse_type(self):
        """Generate parsing type for proto example deserialization.
        
        Returns:
          Parsing type for proto example deserialization.
        """
        return tf.io.FixedLenFeature(self.shape, self.dtype, self.default)

    @property
    def name(self):
        """Return feature name.
        
        Returns:
          Feature name applied with replacement pattern.
        """
        name_str = self._name
        if isinstance(self.replace, dict):
            for k, v in self.replace.items():
                name_str = name_str.replace(k, v)
        return name_str


class Int64Feature(Feature):
    """Class for int64 Feature
    
    Args:
      name: feature name.
      shape: data shape of input.
      default: default value for serialization if value is None or value to be used if an example is missing 
      in deserialization.
      replace: a dictionary in which a string key is replaced by the corresponding value.   
    """
    def __init__(self, name="Int64Feature", shape=[], default=-1, replace=None):
        super(Int64Feature, self).__init__(name=name, dtype=tf.int64, shape=shape, default=default, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature.int64_feature(value)


class Float32Feature(Feature):
    """Class for float32 Feature
    
    Args:
      name: feature name.
      shape: data shape of input.
      default: default value for serialization if value is None or value to be used if an example is missing 
      in deserialization.
      replace: a dictionary in which a string key is replaced by the corresponding value.   
    """
    def __init__(self, name="Float32Feature", shape=[], default=-1, replace=None):
        super(Float32Feature, self).__init__(name=name, dtype=tf.float32, shape=shape, default=default, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature.float_feature(value)


class BytesFeature(Feature):
    """Class for bytes Feature
    
    Args:
      name: feature name.
      shape: data shape of input.
      default: default value for serialization if value is None or value to be used if an example is missing 
      in deserialization.
      replace: a dictionary in which a string key is replaced by the corresponding value.   
    """
    def __init__(self, name="BytesFeature", shape=[], default="", replace=None):
        super(BytesFeature, self).__init__(name=name, dtype=tf.string, shape=shape, default=default, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature.bytes_feature(value)


class SparseFeature(Feature):
    """Base class for Sparse Feature
    
    Args:
      name: feature name.
      dtype: data type of input.
      replace: a dictionary in which a string key is replaced by the corresponding value.   
    """
    def __init__(self, name, dtype, replace=None):
        super(SparseFeature, self).__init__(name=name, dtype=dtype, replace=replace)

    @property
    def parse_type(self):
        return tf.io.VarLenFeature(self.dtype)


class SparseInt64Feature(SparseFeature):
    def __init__(self, name="SparseInt64Feature", replace=None):
        super(SparseInt64Feature, self).__init__(name=name, dtype=tf.int64, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature.int64_feature(value)


class SparseFloat32Feature(SparseFeature):
    def __init__(self, name="SparseFloat32Feature", replace=None):
        super(SparseFloat32Feature, self).__init__(name=name, dtype=tf.float32, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature.float_feature(value)


class SparseBytesFeature(SparseFeature):
    def __init__(self, name="SparseBytesFeature", replace=None):
        super(SparseBytesFeature, self).__init__(name=name, dtype=tf.string, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature.bytes_feature(value)


class FeatureList(object):
    def __init__(self, name, dtype, shape=[], allow_missing=True, replace=None):
        self._name = name
        self.dtype = dtype
        self.shape = shape
        self.allow_missing = allow_missing
        # self.default = default
        self.replace = replace

    @staticmethod
    def int64_feature(values):
        """Wrapper for inserting int64 features into."""
        return tf.train.FeatureList(feature=[Feature.int64_feature(value) for value in values])

    @staticmethod
    def float_feature(values):
        """Wrapper for inserting float features into"""
        return tf.train.FeatureList(feature=[Feature.float_feature(value) for value in values])

    @staticmethod
    def bytes_feature(values):
        """Wrapper for inserting bytes features into"""
        return tf.train.FeatureList(feature=[Feature.bytes_feature(value) for value in values])

    @staticmethod
    def _encode(value):
        raise NotImplementedError

    def __call__(self, value=None):
        # if value is None:
        #     value = self.default
        return self.name, self._encode(value)

    @property
    def parse_type(self):
        return tf.io.FixedLenSequenceFeature(self.shape, self.dtype, self.allow_missing)

    @property
    def name(self):
        name_str = self._name
        if isinstance(self.replace, dict):
            for k, v in self.replace.items():
                name_str = name_str.replace(k, v)
        return name_str


class Int64FeatureList(FeatureList):
    def __init__(self, name="Int64FeatureList", shape=[], allow_missing=True, replace=None):
        super(Int64FeatureList, self).__init__(name=name, dtype=tf.int64, shape=shape, allow_missing=allow_missing,
                                               replace=replace)

    @staticmethod
    def _encode(values):
        return FeatureList.int64_feature(values)


class Float32FeatureList(FeatureList):
    def __init__(self, name="Float32FeatureList", shape=[], allow_missing=True, replace=None):
        super(Float32FeatureList, self).__init__(name=name, dtype=tf.float32, shape=shape, allow_missing=allow_missing,
                                                 replace=replace)

    @staticmethod
    def _encode(values):
        return FeatureList.float_feature(values)


class BytesFeatureList(FeatureList):
    def __init__(self, name="BytesFeatureList", shape=[], allow_missing=True, replace=None):
        super(BytesFeatureList, self).__init__(name=name, dtype=tf.string, shape=shape, allow_missing=allow_missing,
                                               replace=replace)

    @staticmethod
    def _encode(values):
        return FeatureList.bytes_feature(values)


class FeaturesMeta(type):
    ORDER = "_order"

    def __new__(mcs, clsname, bases, clsdict):
        d = dict(clsdict)
        order = []
        for base in reversed(bases):
            if issubclass(base, Features) and base is not Features \
                    or issubclass(base, SequenceFeatures) and base is not SequenceFeatures:
                base_order = base.__dict__.get(mcs.ORDER)
                for name in base_order:
                    val = base.__dict__.get(name)
                    d[name] = val
                order = base_order + order
        for name, value in clsdict.items():
            if isinstance(value, Feature) or isinstance(value, FeatureList):
                value._name = name
                order.append(name)
        d[mcs.ORDER] = order
        return type.__new__(mcs, clsname, bases, d)

    @classmethod
    def __prepare__(mcs, name, bases):
        return OrderedDict()


class Features(six.with_metaclass(FeaturesMeta)):
    @classmethod
    def all_feature_names(cls):
        return list(cls.__dict__.get(cls.ORDER))

    @classmethod
    def feature_map(cls):
        d = OrderedDict()
        for name in cls.__dict__.get(cls.ORDER):
            feature = cls.__dict__.get(name)
            d[feature.name] = feature.parse_type
        return d

    @classmethod
    def to_tf_feature(cls, fea, value):
        """
        TODO: remove
        """
        if isinstance(fea, Feature):
            tf_feature = fea(value)
        elif isinstance(fea, str):
            feaObj = cls.__dict__.get(fea)
            tf_feature = feaObj(value)
        else:
            raise TypeError("feature")
        return tf_feature

    @classmethod
    def to_pb_features(cls, feature_tuples):
        return tf.train.Features(feature=dict(feature_tuples))

    @classmethod
    def to_pb_example(cls, feature_tuples):
        features = cls.to_pb_features(feature_tuples)
        return tf.train.Example(features=features)


class SequenceFeatures(six.with_metaclass(FeaturesMeta)):
    @classmethod
    def all_feature_names(cls):
        return list(cls.__dict__.get(cls.ORDER))

    @classmethod
    def context_feature_names(cls):
        context_features = [name for name in cls.__dict__.get(cls.ORDER)
                            if isinstance(cls.__dict__.get(name), Feature)]
        return context_features

    @classmethod
    def feature_list_names(cls):
        feature_lists = [name for name in cls.__dict__.get(cls.ORDER)
                         if isinstance(cls.__dict__.get(name), FeatureList)]
        return feature_lists

    @classmethod
    def context_feature_map(cls):
        d = OrderedDict()
        for name in cls.context_feature_names():
            feature = cls.__dict__.get(name)
            d[feature.name] = feature.parse_type
        return d

    @classmethod
    def feature_list_map(cls):
        d = OrderedDict()
        for name in cls.feature_list_names():
            feature = cls.__dict__.get(name)
            d[feature.name] = feature.parse_type
        return d

    @classmethod
    def to_pb_features(cls, feature_tuples):
        return tf.train.Features(feature=dict(feature_tuples))

    @classmethod
    def to_pb_feature_lists(cls, feature_list_tuples):
        return tf.train.FeatureLists(feature_list=dict(feature_list_tuples))

    @classmethod
    def to_pb_sequence_example(cls, feature_tuples):
        kwargs = dict()
        context = list()
        feature_lists = list()
        for name, value in feature_tuples:
            if isinstance(cls.__dict__.get(name), Feature):
                context.append((name, value))
            else:
                feature_lists.append((name, value))
        if context:
            kwargs["context"] = cls.to_pb_features(context)
        if feature_lists:
            kwargs["feature_lists"] = cls.to_pb_feature_lists(feature_lists)
        return tf.train.SequenceExample(**kwargs)
