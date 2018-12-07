# Author: Jessy Li (jessy@austin.utexas.edu)

from collections import defaultdict
from rstnode import parse_gcrf_output

class RSTDependency(object):
    """A dependency tree representation of an RST parse.
    """
    def __init__(self, rstroot):
        # List of EDUs
        self.nodes = rstroot.get_leaves()

        # Mapping of EDU to its ID.
        self.node_to_id = {b: a for a, b in enumerate(self.nodes)}

        # List of edges
        self.edges = []

        # <eduid,[ancestor eduids]>; <eduid,-1> if root
        self.ancestors = defaultdict(list)

        rootpromo = rstroot.promotion_set()  # Promotion set of root
        for i, node in enumerate(self.nodes):
            if node in rootpromo:
                self.edges.append(Edge(i, None, None))
            else:
                self.edges.append(self.add_nucleus_edge(i, node, rstroot))

    def add_nucleus_edge(self, idx, node, root):
        """Attach to the closest ancestor for which this node is not in
        the promotion set.
        """
        curp = node.parent
        while (node in curp.promotion_set()):
            curp = curp.parent

        if curp is None:
            # FIXME: logging
            print "ERROR wrong node reached root promotion set"
            return
        tgtnode = curp.promotion_set()[0]
        relation = curp.label
        return Edge(idx, self.node_to_id[tgtnode], relation)

    def get_ancestor(self, node):
        """Get index of ancestors; -1 if root
        """
        idx = self.node_to_id[node]
        return self._get_ancestor_given_idx(idx)

    def _get_ancestor_given_idx(self, idx):
        """Helper function for get_ancestor().
        """
        if self.edges[idx].tgt_idx is None:  # root
            return -1
        if idx not in self.ancestors:
            edge = self.edges[idx]
            while edge.tgt_idx is not None:
                self.ancestors[idx].append(edge.tgt_idx)
                edge = self.edges[edge.tgt_idx]
                if len(self.ancestors[idx]) > len(self.edges):
                    # print "node ",idx," not well formed!"
                    self.ancestors[idx] = None
                    return
        return self.ancestors[idx]

    def validate(self):
        """Return whether this is a well-formed tree.
        """
        for i in range(len(self.nodes)):
            if self._get_ancestor_given_idx(i) is None:
                return False
        return True

    def is_projective(self):
        for edge1 in self.edges:
            for edge2 in self.edges:
                if edge1.src_idx == edge2.src_idx:
                    # Ignore identical arcs and arcs which share a head
                    continue
                sm1, lg1 = (edge1.src_idx, edge1.tgt_idx) \
                    if edge1.src_idx < edge1.tgt_idx \
                    else (edge1.tgt_idx, edge1.src_idx)
                sm2, lg2 = (edge2.src_idx, edge2.tgt_idx) \
                    if edge2.src_idx < edge2.tgt_idx \
                    else (edge2.tgt_idx, edge2.src_idx)
                if sm1 < sm2 < lg1 < lg2 or sm2 < sm1 < lg2 < lg1:
                     return False
        return True


class Edge(object):

    def __init__(self, srcidx, tgtidx, relation):
        self.src_idx = srcidx
        self.tgt_idx = tgtidx
        self.relation = relation

    def __repr__(self):
        return str(self.src_idx)+"-"+str(self.tgt_idx)+","+str(self.relation)


def collapse_edus(rst_node):
	leaves = rst_node.get_leaves()
	for leaf in reversed(leaves):
		children = leaf.parent.children
		if children:
		  if not children[0].eos and children[1].eos:
			edu_text = ' '.join([children[0].label, children[1].label])
			parent = leaf.parent
			parent.label = edu_text
			parent.children = None
			parent.children_nuc = None
			parent.eos = True
			

if __name__ == "__main__":
	testfile = "test.tree"
	rst = parse_gcrf_output(open(testfile).read())
	print rst
	collapse_edus(rst)
	print("After", rst) 
	rstdep = RSTDependency(rst)
	print rstdep.edges
	print rstdep.validate()
	print rstdep.is_projective()