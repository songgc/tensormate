import copy

import numpy as np
import tensorflow as tf
from tensorflow.contrib import graph_editor as ge
from tensorflow.core.framework import graph_pb2


class TfGgraphBuilder(object):

    def __init__(self, scope=None, device=None):
        self._call_count = 0
        self._scope = scope
        self._shapes = []
        self._device = device

    def _build(self, *args, **kwargs):
        raise NotImplementedError("Please implement this method")

    def subgraph(self):
        sgv = ge.sgv_scope(self.scope, tf.get_default_graph())
        inputs_ = [node.name.split(":")[0] for node in sgv.inputs]
        outputs_ = [node.name.split(":")[0] for node in sgv.outputs]
        out = _extract_sub_graph(tf.get_default_graph().as_graph_def(), outputs_, inputs_, self.scope)
        return out

    def visualize(self, output_file=None, whole_graph=False):
        """Visualize TensorFlow graph."""
        if self.ref_count == 0:
            raise RuntimeError("Not built yet")
        if whole_graph:
            graph = tf.get_default_graph()
            graph_def = graph.as_graph_def(add_shapes=True)
        else:
            graph_def = self.subgraph()
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

    def __call__(self, *args, **kwargs):
        is_training = kwargs.get("is_training", True)
        reuse = self.ref_count > 0 and not is_training
        with tf.variable_scope(self.scope, reuse=reuse):
            with tf.device(self._device):
                output = self._build(*args, **kwargs)
            self._call_count += 1
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

    def _infer_output_shape(self, tensor):
        assert tf.is_numeric_tensor(tensor)
        self._shapes.append((tensor.name, tensor.get_shape().as_list()))

    def get_shapes(self):
        if self.ref_count == 0:
            raise RuntimeError("Not built yet")
        return self._shapes

    def get_trainable_variables(self):
        if self.ref_count == 0:
            raise RuntimeError("Not built yet")
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def get_update_ops(self):
        if self.ref_count == 0:
            raise RuntimeError("Not built yet")
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope)

    def get_model_info(self):
        objs = self.get_trainable_variables()
        output = []
        for obj in objs:
            output.append(obj.name)
        return output

    def count_on_conditions(self, strs):
        pass


def _node_name(n):
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]


def _extract_sub_graph(graph_def, dest_nodes, input_nodes, scope):
    """Extract the subgraph that can reach any of the nodes in 'dest_nodes'.

    Args:
      graph_def: A graph_pb2.GraphDef proto.
      dest_nodes: A list of strings specifying the destination node names.
    Returns:
      The GraphDef of the sub-graph.

    Raises:
      TypeError: If 'graph_def' is not a graph_pb2.GraphDef proto.
    """

    # if not isinstance(graph_def, graph_pb2.GraphDef):
    #     raise TypeError("graph_def must be a graph_pb2.GraphDef proto.")

    edges = {}  # Keyed by the dest node name.
    name_to_node_map = {}  # Keyed by node name.

    # Keeps track of node sequences. It is important to still output the
    # operations in the original order.
    node_seq = {}  # Keyed by node name.
    seq = 0
    for node in graph_def.node:
        n = _node_name(node.name)
        name_to_node_map[n] = node
        edges[n] = [_node_name(x) for x in node.input]
        node_seq[n] = seq
        seq += 1
    for d in dest_nodes:
        assert d in name_to_node_map, "%s is not in graph" % d

    nodes_to_keep = set()
    # Breadth first search to find all the nodes that we should keep.
    next_to_visit = dest_nodes[:]
    while next_to_visit:
        n = next_to_visit[0]
        del next_to_visit[0]
        if n in nodes_to_keep:
            # Already visited this node.
            continue
        nodes_to_keep.add(n)
        next_to_visit += edges[n]

    nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
    # Now construct the output GraphDef
    out = graph_pb2.GraphDef()
    for n in nodes_to_keep_list:
        out.node.extend([copy.deepcopy(name_to_node_map[n])])

    return out
