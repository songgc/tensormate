import types
from functools import wraps
import tensorflow as tf
from pprint import pprint


def auto_reuse(scope=""):
    def _wrapper(func):
        return _AutoReuse(func, scope)
    return _wrapper


def shape_info(cached=False):
    def _wrapper(func):
        return _ShapeInfo(func, cached)
    return _wrapper


def op_info(cached=False):
    def _wrapper(func):
        return _OpInfo(func, cached)
    return _wrapper


def _find_input_tensors(args, kwargs):
    tensors = [arg for arg in args if isinstance(arg, (tf.Tensor, tf.Variable))]
    tensors_kw = [arg for arg in kwargs.values() if isinstance(arg, (tf.Tensor, tf.Variable))]
    tensors.extend(tensors_kw)
    return tensors


def _node_name(n):
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]


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
        tensors = _find_input_tensors(args, kwargs)
        if self._cached:
            self._add_to_table(tensors, "inputs")
        else:
            _ShapeInfo._log_shape(tensors, "INPUTS: ")

    def _after_call(self, output):
        tensors = output if isinstance(output, (tuple, list)) else [output]
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

    def clear(self):
        self._result.clear()


class _OpInfo(_GraphDecoratorBase):
    def __init__(self, func, cached):
        super(_OpInfo, self).__init__(func)
        self._cached = cached
        self._input_tensors = None

    def _before_call(self, *args, **kwargs):
        self._input_tensors = _find_input_tensors(args, kwargs)

    def _after_call(self, output):
        input_node_names = [_node_name(node.name) for node in self._input_tensors]
        output_nodes = output if isinstance(output, (tuple, list)) else [output]
        dest_node_names = [_node_name(node.name) for node in output_nodes]

        graph = tf.get_default_graph()
        subgraph = SubGraph(name_scope=tf.contrib.framework.get_name_scope(), graph=graph)
        pprint(input_node_names)
        subgraph_nodes = subgraph.extract_subgraph_nodes(dest_node_names)
        before_input_nodes = subgraph.extract_subgraph_nodes(input_node_names)
        subgraph_nodes = [node for node in subgraph_nodes if node not in before_input_nodes or node in input_node_names]

        if self._cached:
            pass
        else:
            for node in subgraph_nodes:
                tf.logging.info(node + " " + str(_OpInfo.get_output_shapes_by_node_name(node))
                                + " from " + str(subgraph.edges(node)))

    @staticmethod
    def get_output_shapes_by_node_name(node_name, graph=None):
        graph = tf.get_default_graph() if graph is None else graph
        outputs = graph.get_operation_by_name(node_name).outputs
        shapes = [output.get_shape().as_list() for output in outputs]
        return tuple(shapes) if len(shapes) > 1 else shapes[0]


class SubGraph(object):
    def __init__(self, name_scope=None, graph=None):
        self._name_scope = "" if name_scope is None else name_scope
        self._graph = tf.get_default_graph() if graph is None else graph

        edges = {}  # Keyed by the dest node name.
        name_to_node_map = {}  # Keyed by node name.
        # name_scope = tf.contrib.framework.get_name_scope()
        node_seq = {}  # Keyed by node name.
        seq = 0
        # graph = tf.get_default_graph()
        graph_def = self._graph.as_graph_def()
        for node in graph_def.node:
            n = _node_name(node.name)
            if self._name_scope not in n:
                # and n not in input_node_names:
                # print(n)
                continue
            name_to_node_map[n] = node
            edges[n] = [_node_name(x) for x in node.input]
            node_seq[n] = seq
            seq += 1

        self._edges = edges
        self._name_to_node_map = name_to_node_map
        self._node_seq = node_seq

    @property
    def name_scope(self):
        return self._name_scape

    @property
    def graph(self):
        return self._graph

    def edges(self, node_name):
        return self._edges[node_name]

    def node_by_name(self, node_name):
        return self._name_to_node_map[node_name]

    def seq_by_name(self, node_name):
        return self._node_seq[node_name]

    def extract_subgraph_nodes(self, dest_node_names):
        nodes_to_keep = set()
        # Breadth first search to find all the nodes that we should keep.
        next_to_visit = dest_node_names[:]
        while next_to_visit:
            n = next_to_visit[0]
            del next_to_visit[0]
            if n in nodes_to_keep:
                # Already visited this node.
                continue
            nodes_to_keep.add(n)
            next_to_visit += self.edges(n)

        nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: self.seq_by_name(n))
        return nodes_to_keep_list
