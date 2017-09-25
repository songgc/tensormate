import copy
import types
from collections import Counter, deque
from functools import wraps

import numpy as np
import six
import tensorflow as tf
from tensorflow.core.framework import graph_pb2, node_def_pb2
from tensorflow.python.util import compat


def auto_reuse(scope=None):
    def _wrapper(func):
        return _AutoReuse(func, scope)
    return _wrapper


def shape_info(cached=False):
    def _wrapper(func):
        return _ShapeInfo(func, cached)
    return _wrapper


def graph_info(cached=False):
    def _wrapper(func):
        return _GraphInfo(func, cached)
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
    def __init__(self, func, scope=None):
        super(_AutoReuse, self).__init__(func=func)
        self._scope = "" if scope is None else scope
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
            self._add_to_table(tensors, "INPUT")
        else:
            _ShapeInfo._log_shape(tensors, "INPUT")

    def _after_call(self, output):
        tensors = output if isinstance(output, (tuple, list)) else [output]
        if self._cached:
            self._add_to_table(tensors, "OUTPUT")
        else:
            _ShapeInfo._log_shape(tensors, "OUTPUT")

    @staticmethod
    def _log_shape(tensors, io_type):
        for tensor in tensors:
            msg = " {:<40}{:15}{}".format(tensor.name, io_type, str(tensor.get_shape().as_list()))
            tf.logging.info(msg)

    def _add_to_table(self, tensors, io_type):
        for tensor in tensors:
            self._result.append((tensor.name, io_type, tensor.get_shape().as_list()))

    @property
    def result(self):
        return self._result

    def clear(self):
        self._result.clear()


class _GraphInfo(_GraphDecoratorBase):
    def __init__(self, func, cached):
        super(_GraphInfo, self).__init__(func)
        self._cached = cached
        self._id = None
        self._result = []
        self._graph_def = None
        self._input_tensors = None

    @property
    def id(self):
        return self._id

    @property
    def result(self):
        return self._result

    @property
    def graph_def(self):
        return self._graph_def

    @property
    def viz_html_string(self):
        return self._visualize(self._graph_def, output_file=None)

    def clear(self):
        self._id = None
        self._result.clear()
        self._graph_def = None

    def _before_call(self, *args, **kwargs):
        self._input_tensors = _find_input_tensors(args, kwargs)

    def _after_call(self, output):
        input_node_names = [_node_name(node.name) for node in self._input_tensors]
        output_nodes = output if isinstance(output, (tuple, list)) else [output]
        dest_node_names = [_node_name(node.name) for node in output_nodes]
        self._id = str(dest_node_names)

        graph = tf.get_default_graph()
        subgraph = SubGraph(name_scope=tf.contrib.framework.get_name_scope(), graph=graph)
        subgraph_nodes = subgraph.extract_subgraph_nodes(dest_node_names)
        before_input_nodes = subgraph.extract_subgraph_nodes(input_node_names)
        subgraph_nodes = [node for node in subgraph_nodes if node not in before_input_nodes or node in input_node_names]

        if self._cached:
            for node in subgraph_nodes:
                shapes = _GraphInfo.get_output_shapes_by_node_name(node)
                op = subgraph.node_by_name(node).op
                inputs = subgraph.edges(node)
                self._result.append((node, op, shapes, inputs))
            g = subgraph.new_graph(subgraph_nodes, input_node_names)
            self._graph_def = subgraph.strip_consts(g, max_const_size=32)

        else:
            tf.logging.info("------Subgraph for {} ------".format(self.id))
            vars = []
            ops = []
            io_info = []
            for node in subgraph_nodes:
                shapes = _GraphInfo.get_output_shapes_by_node_name(node)
                op = subgraph.node_by_name(node).op
                inputs = subgraph.edges(node)
                if "Variable" in op:
                    num_params = np.prod(shapes)
                    vars.append((node, str(shapes), num_params))
                ops.append(op)
                if node in input_node_names:
                    io_info.append((node, "INPUT", shapes))
                if node in dest_node_names:
                    io_info.append((node, "OUTPUT", shapes))
                fmt = " {:<40}{:15}{:<22}{}"
                msg = fmt.format(node, op, str(shapes), str(inputs))
                tf.logging.info(msg)
            tf.logging.info("------Variables------")
            fmt = " {:<40}{:<22}{}"
            for t in vars:
                tf.logging.info(fmt.format(*t))
            tf.logging.info("------Ops------")
            fmt = " {:<40}{}"
            counter = Counter(ops)
            for t in counter.most_common(len(ops)):
                tf.logging.info(fmt.format(*t))
            tf.logging.info("------IO------")
            fmt = " {:<40}{:15}{}"
            for t in io_info:
                tf.logging.info(fmt.format(*t))
            tf.logging.info("------End for {}------".format(self.id))

    @staticmethod
    def get_output_shapes_by_node_name(node_name, graph=None):
        graph = tf.get_default_graph() if graph is None else graph
        outputs = graph.get_operation_by_name(node_name).outputs
        shapes = [output.get_shape().as_list() for output in outputs]
        return tuple(shapes) if len(shapes) > 1 else shapes[0]

    @staticmethod
    def _visualize(graph_def, output_file=None):
        """Visualize TensorFlow graph."""
        # strip_def = SubGraph.strip_consts(graph_def, max_const_size=32)
        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(graph_def)), id='graph' + str(np.random.rand()))

        iframe = """
            <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))
        if output_file is None:
            return iframe
        with open(output_file, "tw") as f:
            f.write(iframe)


class SubGraph(object):
    def __init__(self, name_scope=None, graph=None):
        self._name_scope = "" if name_scope is None else name_scope
        self._graph = tf.get_default_graph() if graph is None else graph

        # edges = {}  # Keyed by the dest node name.
        # name_to_node_map = {}  # Keyed by node name.
        # node_seq = {}  # Keyed by node name.
        # seq = 0
        # graph_def = self._graph.as_graph_def()
        # for node in graph_def.node:
        #     n = _node_name(node.name)
        #     if self._name_scope not in n:
        #         # and n not in input_node_names:
        #         # print(n)
        #         continue
        #     name_to_node_map[n] = node
        #     edges[n] = [_node_name(x) for x in node.input]
        #     node_seq[n] = seq
        #     seq += 1
        #
        # self._edges = edges
        # self._name_to_node_map = name_to_node_map
        # self._node_seq = node_seq

    @property
    def name_scope(self):
        return self._name_scope

    @property
    def graph(self):
        return self._graph

    def edges(self, node_name):
        # return self._edges[node_name]
        op = self.graph.get_operation_by_name(node_name)
        return op.node_def.input

    def node_by_name(self, node_name):
        # return self._name_to_node_map[node_name]
        op = self.graph.get_operation_by_name(node_name)
        return op.node_def

    def seq_by_name(self, node_name):
        # return self._node_seq[node_name]
        op = self.graph.get_operation_by_name(node_name)
        return op._id

    def extract_subgraph_nodes(self, dest_node_names):
        nodes_to_keep = set()
        # Breadth first search to find all the nodes that we should keep.
        # next_to_visit = dest_node_names[:]
        next_to_visit = deque(dest_node_names[:])
        while next_to_visit:
            # n = next_to_visit[0]
            # del next_to_visit[0]
            n = next_to_visit.popleft()
            if n in nodes_to_keep:
                # Already visited this node.
                continue
            nodes_to_keep.add(n)
            next_to_visit.extend(self.edges(n))

        nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: self.seq_by_name(n))
        return nodes_to_keep_list

    def new_graph(self, node_names, input_names):
        out_graph = graph_pb2.GraphDef()
        for node_name in node_names:
            if node_name in input_names:
                continue
            node = self.node_by_name(node_name)
            out_graph.node.extend([copy.deepcopy(node)])
            op = self.graph.get_operation_by_name(node.name)
            if op.outputs:
                out_graph.node[-1].attr["_output_shapes"].list.shape.extend([
                    output.get_shape().as_proto() for output in op.outputs])

        for name in input_names:
            op = self.graph.get_operation_by_name(name)
            node = SubGraph._node_def("Placeholder", name)
            out_graph.node.extend([node])
            if op.outputs:
                out_graph.node[-1].attr["_output_shapes"].list.shape.extend([
                    output.get_shape().as_proto() for output in op.outputs])
        return out_graph

    @staticmethod
    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = str.encode("<stripped %s bytes>" % size)
        return strip_def

    @staticmethod
    def _node_def(op_type, name, device=None, attrs=None):
        """Create a NodeDef proto.

        Args:
          op_type: Value for the "op" attribute of the NodeDef proto.
          name: Value for the "name" attribute of the NodeDef proto.
          device: string, device, or function from NodeDef to string.
            Value for the "device" attribute of the NodeDef proto.
          attrs: Optional dictionary where the key is the attribute name (a string)
            and the value is the respective "attr" attribute of the NodeDef proto (an
            AttrValue).

        Returns:
          A node_def_pb2.NodeDef protocol buffer.
        """
        node_def = node_def_pb2.NodeDef()
        node_def.op = compat.as_bytes(op_type)
        node_def.name = compat.as_bytes(name)
        if attrs is not None:
            for k, v in six.iteritems(attrs):
                node_def.attr[k].CopyFrom(v)
        # if device is not None:
        #     if callable(device):
        #         node_def.device = device(node_def)
        #     else:
        #         node_def.device = _device_string(device)
        return node_def

