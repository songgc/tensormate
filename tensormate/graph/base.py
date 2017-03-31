import copy
from collections import Counter

import numpy as np
import six
import tensorflow as tf
from tensorflow.core.framework import graph_pb2, node_def_pb2
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated


class TfGgraphBuilder(object):

    def __init__(self, scope=None, device=None):
        self._call_count = 0
        self._scope = scope
        self._device = device
        self._trainable_variables = None
        self._update_ops = None
        self._shapes = []
        self._created_nodes = []
        self._node_map = dict()
        self._before_states = dict()
        self._after_states = dict()
        self._actual_scopes = []

    def _build(self, *args, **kwargs):
        raise NotImplementedError("Please implement this method")

    def _subgraph(self):
        out_graph = graph_pb2.GraphDef()
        to_be_inputed = []
        for node in self._created_nodes:
            out_graph.node.extend([copy.deepcopy(node)])
            op = tf.get_default_graph().get_operation_by_name(node.name)
            if op.outputs:
                out_graph.node[-1].attr["_output_shapes"].list.shape.extend([
                    output.get_shape().as_proto() for output in op.outputs])
            for name in node.input:
                if "/" not in name:
                    to_be_inputed.append(name)
                else:
                    flag = False
                    for scope in self._actual_scopes:
                        seq = scope.split("/")
                        if "/".join(name.split("/")[0: len(seq)]) == scope:
                            flag = True
                            break
                    if not flag:
                        to_be_inputed.append(name)
                # elif name.split("/")[0] != self.scope:
                #     to_be_inputed.append(name)
        for name in to_be_inputed:
            op = tf.get_default_graph().get_operation_by_name(name)
            node = _NodeDef("Placeholder", name)
            out_graph.node.extend([node])
            if op.outputs:
                out_graph.node[-1].attr["_output_shapes"].list.shape.extend([
                    output.get_shape().as_proto() for output in op.outputs])
        return out_graph

    def visualize(self, output_file=None, whole_graph=False):
        """Visualize TensorFlow graph."""
        if self.ref_count == 0:
            raise RuntimeError("Not built yet")
        if whole_graph:
            graph = tf.get_default_graph()
            graph_def = graph.as_graph_def(add_shapes=True)
        else:
            graph_def = self._subgraph()
        strip_def = self.strip_consts(graph_def, max_const_size=32)
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
        """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

        iframe = """
            <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))
        if output_file is None:
            return iframe
        with open(output_file, "tw") as f:
            f.write(iframe)

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

    def _before_call(self):
        g = tf.get_default_graph().as_graph_def()
        existing_nodes = set([node.name for node in g.node])
        self._before_states = dict()
        self._before_states["existing_nodes"] = existing_nodes
        return

    def _call_body(self, *args, **kwargs):
        # is_training = kwargs.get("is_training", True)
        reuse = self.ref_count > 0
        with tf.variable_scope(self._scope, reuse=reuse):
            if self._device is None:
                output = self._build(*args, **kwargs)
            else:
                with tf.device(self._device):
                    output = self._build(*args, **kwargs)
            scope_name = tf.get_variable_scope().name
            if self.ref_count > 0:
                scope_name += "_" + str(self.ref_count)
            self._actual_scopes.append(scope_name)
        return output

    def _after_call(self):
        existing_nodes = self._before_states["existing_nodes"]
        if self._call_count == 1:
            self._trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope)
        g = tf.get_default_graph().as_graph_def()
        self._created_nodes += [node for node in g.node if node.name not in existing_nodes]

    def __call__(self, *args, **kwargs):

        self._before_call()

        output = self._call_body(*args, **kwargs)

        self._call_count += 1

        self._after_call()

        return output

    @property
    def ref_count(self):
        return self._call_count

    @property
    def scope(self):
        return self._scope

    @property
    def device(self):
        return self._device

    @deprecated("2017-10-31", "Use infer_output_shape(tensor)")
    def _infer_output_shape(self, tensor):
        self.infer_output_shape(tensor)

    def infer_output_shape(self, tensor):
        assert tf.is_numeric_tensor(tensor)
        self._shapes.append((tensor.name, tensor.get_shape().as_list()))

    def get_shapes(self):
        if self.ref_count == 0:
            raise RuntimeError("Not built yet")
        return self._shapes

    def get_trainable_variables(self):
        if self.ref_count == 0:
            raise RuntimeError("Not built yet")
        return self._trainable_variables

    def get_update_ops(self):
        if self.ref_count == 0:
            raise RuntimeError("Not built yet")
        return self._update_ops

    def get_model_info(self):
        objs = self.get_trainable_variables()
        output = []
        for obj in objs:
            output.append(obj.name)
        return output

    def op_counting(self):
        op_list = [node.op for node in self._created_nodes]
        counter = Counter(op_list)
        return counter.most_common(len(op_list))

    def count_on_conditions(self, strs):
        pass

    def add_node_to_map(self, name, node):
        self._node_map[name] = node

    def get_node_from_map(self, name):
        return self._node_map.get(name)

    def get_last_actual_scope(self):
        if self.ref_count == 0:
            raise RuntimeError("Not built yet")
        return self._actual_scopes[-1]


def _node_name(n):
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]


def _NodeDef(op_type, name, device=None, attrs=None):
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
