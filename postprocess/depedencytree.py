from mst import PyMaxSpanTree
import numpy as np
from .. import data_structure


class DependencyTree(object):
    def __init__(self):
        self.edges = []
        self.node_depths = {}
        self.deps = {}
        self.height = None

    def parse_tree(self, mst_tree):
        for pidx, cidx, weight in mst_tree:
            pidx = int(pidx)
            cidx = int(cidx)
            self.edges.append(Edge(pidx, cidx, weight))
            if pidx in self.deps:
                self.deps[pidx].append(cidx)
            else:
                self.deps[pidx] = [cidx]

            self._set_height()
            self._set_node_depths()

    def _set_height(self):
        self.height = len(self.deps.keys())

    def _set_node_depths(self):
        for parent_num, parent in enumerate(self.deps.keys()):
            self.node_depths[parent_num] = len(self.deps[parent])


class Edge(object):
    def __init__(self, srcidx, tgtidx, weight):
        self.src_idx = srcidx
        self.tgt_idx = tgtidx
        self.weight = weight

    def __repr__(self):
        return str(self.src_idx)+"-"+str(self.tgt_idx)+","+str(self.weight)


def set_tree(docs):
    mst_obj = PyMaxSpanTree()
    for doc in docs:
        dep_tree = DependencyTree()
        str_scores = np.delete(doc.str_scores, 0, 0)  # delete first row for ROOT being the child
        mst_tree = mst_obj.get_tree(str_scores)
        dep_tree.parse_tree(mst_tree)











