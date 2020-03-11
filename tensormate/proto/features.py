from collections import OrderedDict
from functools import partial

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
    def __init__(self, name, dtype, shape=(), default=None, replace=None):
        self._name = name
        self.dtype = dtype
        self.shape = shape
        self.default = default
        self.replace = replace

    @staticmethod
    def _int64_feature(value):
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
    def _float_feature(value):
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
    def _bytes_feature(value):
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

    # TODO change pars_type to parsing_type
    def parsing_type(self):
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

    def __str__(self):
        string = "{}(name={}, dtype={}, shape={}, default={}, replace={})"\
            .format(self.__class__.__name__, self.name, self.dtype, self.shape, self.default, self.replace)
        return string


class Int64Feature(Feature):
    """Class for int64 Feature
    
    Args:
      name: feature name.
      shape: data shape of input.
      default: default value for serialization if value is None or value to be used if an example is missing 
      in deserialization.
      replace: a dictionary in which a string key is replaced by the corresponding value.   
    """
    def __init__(self, name="Int64Feature", shape=(), default=-1, replace=None):
        super(Int64Feature, self).__init__(name=name, dtype=tf.int64, shape=shape, default=default, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature._int64_feature(value)


class Float32Feature(Feature):
    """Class for float32 Feature
    
    Args:
      name: feature name.
      shape: data shape of input.
      default: default value for serialization if value is None or value to be used if an example is missing 
      in deserialization.
      replace: a dictionary in which a string key is replaced by the corresponding value.   
    """
    def __init__(self, name="Float32Feature", shape=(), default=-1, replace=None):
        super(Float32Feature, self).__init__(name=name, dtype=tf.float32, shape=shape, default=default, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature._float_feature(value)


class BytesFeature(Feature):
    """Class for bytes Feature
    
    Args:
      name: feature name.
      shape: data shape of input.
      default: default value for serialization if value is None or value to be used if an example is missing 
      in deserialization.
      replace: a dictionary in which a string key is replaced by the corresponding value.   
    """
    def __init__(self, name="BytesFeature", shape=(), default="", replace=None):
        super(BytesFeature, self).__init__(name=name, dtype=tf.string, shape=shape, default=default, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature._bytes_feature(value)


class SparseFeature(Feature):
    """Base class for Sparse Feature
    
    Args:
      name: feature name.
      dtype: data type of input.
      replace: a dictionary in which a string key is replaced by the corresponding value.   
    """
    def __init__(self, name, dtype, replace=None):
        super(SparseFeature, self).__init__(name=name, dtype=dtype, replace=replace)

    def parsing_type(self):
        return tf.io.VarLenFeature(self.dtype)


class SparseInt64Feature(SparseFeature):
    def __init__(self, name="SparseInt64Feature", replace=None):
        super(SparseInt64Feature, self).__init__(name=name, dtype=tf.int64, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature._int64_feature(value)


class SparseFloat32Feature(SparseFeature):
    def __init__(self, name="SparseFloat32Feature", replace=None):
        super(SparseFloat32Feature, self).__init__(name=name, dtype=tf.float32, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature._float_feature(value)


class SparseBytesFeature(SparseFeature):
    def __init__(self, name="SparseBytesFeature", replace=None):
        super(SparseBytesFeature, self).__init__(name=name, dtype=tf.string, replace=replace)

    @staticmethod
    def _encode(value):
        return Feature._bytes_feature(value)


class FeatureList(object):
    def __init__(self, name, dtype, shape=(), allow_missing=True, replace=None):
        self._name = name
        self.dtype = dtype
        self.shape = shape
        self.allow_missing = allow_missing
        # self.default = default
        self.replace = replace

    @staticmethod
    def _int64_feature(values):
        """Wrapper for inserting int64 features into."""
        return tf.train.FeatureList(feature=[Feature._int64_feature(value) for value in values])

    @staticmethod
    def _float_feature(values):
        """Wrapper for inserting float features into"""
        return tf.train.FeatureList(feature=[Feature._float_feature(value) for value in values])

    @staticmethod
    def _bytes_feature(values):
        """Wrapper for inserting bytes features into"""
        return tf.train.FeatureList(feature=[Feature._bytes_feature(value) for value in values])

    @staticmethod
    def _encode(value):
        raise NotImplementedError

    def __call__(self, value=None):
        # if value is None:
        #     value = self.default
        return self.name, self._encode(value)

    def parsing_type(self, target="ragged"):
        """

        :param target: "ragged" or "fixed_len"
        :return:
        """
        if target == "ragged":
            out = tf.io.RaggedFeature(self.dtype)
        else:
            out = tf.io.FixedLenSequenceFeature(self.shape, self.dtype, self.allow_missing)
        return out

    @property
    def name(self):
        name_str = self._name
        if isinstance(self.replace, dict):
            for k, v in self.replace.items():
                name_str = name_str.replace(k, v)
        return name_str

    def __str__(self):
        string = "{}(name={}, dtype={}, shape={}, allow_missing={}, replace={})"\
            .format(self.__class__.__name__,
                    self.name, self.dtype, self.shape, self.allow_missing, self.replace)
        return string


class Int64FeatureList(FeatureList):
    def __init__(self, name="Int64FeatureList", shape=(), allow_missing=True, replace=None):
        super(Int64FeatureList, self).__init__(name=name, dtype=tf.int64, shape=shape, allow_missing=allow_missing,
                                               replace=replace)

    @staticmethod
    def _encode(values):
        return FeatureList._int64_feature(values)


class Float32FeatureList(FeatureList):
    def __init__(self, name="Float32FeatureList", shape=(), allow_missing=True, replace=None):
        super(Float32FeatureList, self).__init__(name=name, dtype=tf.float32, shape=shape, allow_missing=allow_missing,
                                                 replace=replace)

    @staticmethod
    def _encode(values):
        return FeatureList._float_feature(values)


class BytesFeatureList(FeatureList):
    def __init__(self, name="BytesFeatureList", shape=(), allow_missing=True, replace=None):
        super(BytesFeatureList, self).__init__(name=name, dtype=tf.string, shape=shape, allow_missing=allow_missing,
                                               replace=replace)

    @staticmethod
    def _encode(values):
        return FeatureList._bytes_feature(values)


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

    def __getitem__(cls, name):
        return cls.__dict__[name]

    def __str__(cls):
        name = "{}".format(cls.__mro__[0].__name__)
        strs = list()
        strs.append(name + ":")
        feat_names = cls.all_feature_names(with_replacement=False)
        for fname in feat_names:
            strs.append("  " + str(cls[fname]))
        string = "\n".join(strs)
        return string

    def all_member_names(cls, with_replacement=True):
        if with_replacement:
            output = [cls.__getitem__(key).name for key in cls.__getitem__(cls.ORDER)]
        else:
            output = cls[cls.ORDER]
        return output

    # TODO remove
    def all_feature_names(cls, with_replacement=True):
        return cls.all_member_names(with_replacement)

    def _get_features(cls, dtype, is_sequence, mode):
        """

        :param dtype: "numeric' or "string", "all"
        :param is_sequence: bool
        :param mode: "cls" or "name"
        :return: list of features per request
        """
        all_features = cls.all_feature_names()
        outputs = []
        if dtype == "string":
            dtype_set = (tf.dtypes.string,)
        elif dtype == "numeric":
            dtype_set = (tf.dtypes.int64, tf.dtypes.float32)
        elif dtype == "all":
            dtype_set = (tf.dtypes.int64, tf.dtypes.float32, tf.dtypes.string)
        else:
            raise RuntimeError("Unknown date type")

        base_class = FeatureList if is_sequence else Feature
        for name in all_features:
            feat = cls.__getitem__(name)
            if feat.dtype in dtype_set and isinstance(feat, base_class):
                element = feat if mode == "cls" else feat.name
                outputs.append(element)
        return outputs

    def feature_members(cls):
        return cls._get_features("all", False, "cls")

    def feature_names(cls):
        return cls._get_features("all", False, "name")

    def featurelist_members(cls):
        return cls._get_features("all", True, "cls")

    def featurelist_names(cls):
        return  cls._get_features("all", True, "name")

    def numeric_feature_members(cls):
        return cls._get_features("numeric", False, "cls")

    def numeric_feature_names(cls):
        return cls._get_features("numeric", False, "name")

    def numeric_featurelist_members(cls):
        return cls._get_features("numeric", True, "cls")

    def numeric_featurelist_names(cls):
        return cls._get_features("numeric", True, "name")

    def string_feature_members(cls):
        return cls._get_features("string", False, "cls")

    def string_feature_names(cls):
        return cls._get_features("string", False, "name")

    def string_featurelist_members(cls):
        return cls._get_features("string", True, "cls")

    def string_featurelist_names(cls):
        return cls._get_features("string", True, "name")


class Features(six.with_metaclass(FeaturesMeta)):

    @classmethod
    def feature_map(cls):
        d = OrderedDict()
        for name in cls.__getitem__(cls.ORDER):
            feature = cls.__getitem__(name)
            d[feature.name] = feature.parsing_type()
        return d

    @classmethod
    def to_tf_feature(cls, fea, value):
        """
        TODO: remove
        """
        if isinstance(fea, Feature):
            tf_feature = fea(value)
        elif isinstance(fea, str):
            feaObj = cls.__getitem__(fea)
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

    @classmethod
    def parse_function(cls):
        parser = partial(tf.io.parse_example, features=cls.feature_map())
        return parser


class SequenceFeatures(six.with_metaclass(FeaturesMeta)):

    @classmethod
    def context_feature_names(cls):
        return cls.feature_names()

    @classmethod
    def feature_list_names(cls):
        return cls.featurelist_names()

    @classmethod
    def context_feature_map(cls):
        d = OrderedDict()
        for name in cls.context_feature_names():
            feature = cls.__getitem__(name)
            d[feature.name] = feature.parsing_type()
        return d

    @classmethod
    def feature_list_map(cls, target="ragged"):
        d = OrderedDict()
        for name in cls.feature_list_names():
            feature = cls.__getitem__(name)
            d[feature.name] = feature.parsing_type(target)
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
            if isinstance(cls.__getitem__(name), Feature):
                context.append((name, value))
            else:
                feature_lists.append((name, value))
        if context:
            kwargs["context"] = cls.to_pb_features(context)
        if feature_lists:
            kwargs["feature_lists"] = cls.to_pb_feature_lists(feature_lists)
        return tf.train.SequenceExample(**kwargs)

    @classmethod
    def parser_function(cls, target="ragged", totensor=True):

        def parser(serialized):
            context, rseq, sparse = tf.io.parse_sequence_example(serialized=serialized,
                                                            context_features=cls.context_feature_map(),
                                                            sequence_features=cls.feature_list_map(target))

            if target != "ragged" or not totensor:
                return context, rseq, sparse

            seq = {}
            for k, v in rseq.items():
                seq[k] = v.merge_dims(0, -1)
            return context, seq, sparse

        return parser
