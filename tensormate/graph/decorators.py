import types
from functools import wraps
import tensorflow as tf


def auto_reuse(scope=""):
    def _wrapper(func):
        return _AutoReuse(func, scope)
    return _wrapper


def shape_info(cached=False):
    def _wrapper(func):
        return _ShapeInfo(func, cached)
    return _wrapper


class _GraphDecoratorBase(object):
    def __init__(self, func):
        wraps(func)(self)

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)

    def _before_call(self, *args, **kwargs):
        pass

    def _after_call(self, output):
        pass

    def __call__(self, *args, **kwargs):
        self._before_call(*args, **kwargs)
        output = self.__wrapped__(*args, **kwargs)
        self._after_call(output)
        return output


class _AutoReuse(_GraphDecoratorBase):
    def __init__(self, func, scope=""):
        super(_AutoReuse, self).__init__(func=func)
        self._scope = scope
        self._ref_count = 0

    def __call__(self, *args, **kwargs):
        reuse = self.ref_count > 0
        with tf.variable_scope(self._scope, reuse=reuse):
            output = self.__wrapped__(*args, **kwargs)
        self._ref_count += 1
        return output

    @property
    def scope(self):
        return self._scope

    @property
    def ref_count(self):
        return self._ref_count


class _ShapeInfo(_GraphDecoratorBase):
    def __init__(self, func, cached):
        super(_ShapeInfo, self).__init__(func=func)
        self._cached = cached
        self._result = []

    def _before_call(self, *args, **kwargs):
        tensors = [arg for arg in args if isinstance(arg, (tf.Tensor, tf.Variable))]
        tensors_kw = [arg for arg in kwargs.values() if isinstance(arg, (tf.Tensor, tf.Variable))]
        tensors.extend(tensors_kw)
        if self._cached:
            self._add_to_table(tensors, "inputs")
        else:
            _ShapeInfo._log_shape(tensors, "INPUTS: ")

    def _after_call(self, output):
        if isinstance(output, (tuple, list)):
            tensors = output
        else:
            tensors = [output]
        if self._cached:
            self._add_to_table(tensors, "outputs")
        else:
            _ShapeInfo._log_shape(tensors, "OUTPUTS: ")

    @staticmethod
    def _log_shape(tensors, prefix):
        for tensor in tensors:
            msg = prefix + tensor.name + ": " + str(tensor.get_shape().as_list())
            tf.logging.info(msg)

    def _add_to_table(self, tensors, prefix):
        for tensor in tensors:
            self._result.append((prefix, tensor.name, tensor.get_shape().as_list()))

    @property
    def result(self):
        return self._result

    @property
    def clear(self):
        self._result.clear()


class OpInfo(_GraphDecoratorBase):
    def __init__(self, func, cached=True):
        super(OpInfo, self).__init__(func)
        self._cached = cached


