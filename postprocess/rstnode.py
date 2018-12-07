# Author: Jessy Li (jessy@austin.utexas.edu)

import logging 

def parse_gcrf_output(treestr, dummy=False):
    """Take a string formatted parse (i.e., from gCRF output files)
    and produce the root node.
    """
    if dummy:
        # Create dummy tree with a single dummy root node
        self.root = RSTNode(treestr, dummy=True)
        return

    nodestack = []
    for line in _fix_lines(treestr):
        logging.debug("{0}".format(line))
        line = line.strip()
        
        if line.startswith("("):
            # Regular node
            nodestack.append(RSTNode(line))
            logging.debug("PUSH {0}".format(nodestack[-1].label))

        elif line.startswith("_!"):
            # Leaf / EDU
            leaf = RSTNode(line, leaf=True)
            leaf.parent = nodestack[-1]
            nodestack[-1].children.append(leaf)

            # Traverse up the tree to assign parents and children
            while line.endswith(")") and len(nodestack) > 1:
                nodestack[-1].parent = nodestack[-2]
                nodestack[-2].children.append(nodestack[-1])
                logging.debug("POP {0} ({1} children)"
                              .format(nodestack[-1].label,
                                      len(nodestack[-1].children)))

                nodestack.pop()
                line = line[:-1]
        else:
            logging.warning("Unexpected starting line {0}".format(line))

    assert len(nodestack) == 1
    return nodestack[0]


def _fix_lines(treestr):
    """Fix line boundaries.
    """
    lines = treestr.strip().split("\n")
    retlines = []
    for line in lines:
        if line.strip().startswith("_!"):
            temp = line.strip()
            while temp.endswith(")"):
                temp = temp[:-1]
            if temp.endswith("!_"):
                retlines.append(line.strip())
                continue
        ct = line.count("!_")
        for i, field in enumerate(line.split("_!")):
            if i == 0 and len(field.strip()) > 0:
                retlines.append(field)
            elif i > 0 and ct > 0:
                retlines.append("_!"+field)
                ct -= 1
            else:
                retlines[-1] += field
    return retlines


class RSTNode(object):
    """A node in an RST tree.
    """

    def __init__(self, nodestr, leaf=False, dummy=False):
        self.label = None           # if leaf then EDU text else relation
        self.parent = None          # RSTNode
        self.children = None        # list of RSTNodes
        self.children_nuc = None    # list of "N" or "S"
        self.eop = False            # if end of paragraph
        self.eos = False            # if end of sentence
        self.promoset = None        # Promotion set (Marcu 1999)
        self._height = None         # distance to deepest leaf
        self._depth = None          # distance to root
        self._marcu_score = None    # Marcu score (Marcu 1999)
        self._is_nucleus = None

        # Construct the node from the gCRF serialization
        nodestr = nodestr.strip()
        if dummy:
            # Dummy node containing only label text
            self.label = nodestr
            self.eop = True
            self.eos = True
        elif leaf:
            # Leaf node containing EDU label
            while (nodestr.endswith(")")):
                nodestr = nodestr[:-1]
            self.label = nodestr[2:-2]
            if self.label.endswith("<s>"):
                self.eos = True
                self.label = self.label[:-4]
            elif self.label.endswith("<P>"):
                self.eos = True
                self.eop = True
                self.label = self.label[:-4]
        else:
            # Non-terminal node with two children
            self.label = nodestr[1:-6]
            self.children_nuc = (nodestr[-5], nodestr[-2])
            self.children = []  # will be populated by the parser

    def is_nucleus(self):
        """Return whether the node is a nucleus.
        """
        if self._is_nucleus == None:
            if self.parent is None:
                self._is_nucleus = True
            else:
                idx = 0 if self.parent.children[0] == self else 1
                self._is_nucleus = (self.parent.children_nuc[idx] == 'N')
        return self._is_nucleus

    def is_leaf(self):
        """Whether this node is a leaf.
        """
        return self.children is None

    def get_root(self):
        """Find the root of the discourse tree.
        """
        current = self
        while current.parent is not None:
            current = current.parent
        return current

    def get_nucleus_child(self):
        """Get the nucleus child or children if bi-nuclear.
        """
        if self.is_leaf():
            return None
        return tuple(child
                     for child, label in zip(self.children, self.children_nuc)
                     if label == "N")

    def get_height(self):
        """Return the height of this node in the RST parse.
        """
        if self._height is None:
            self._height = self._height_helper()
        return self._height

    def _height_helper(self):
        if self.is_leaf():
            return 0
        return 1 + max(self.children[0].get_height(), self.children[1].get_height())

    def get_depth(self):
        '''Return the depth of this node in the RST parse.
        '''
        if self._depth is None:
            self._depth = self._depth_helper()
        return self._depth

    def _depth_helper(self):
        if self.parent is None:
            return 0
        return 1 + self.parent._depth_helper()

    def is_descendent(self, node):
        """Return whether this node is a descendent of another.
        """
        if self == node or self.parent == node:
            return True
        elif node.is_leaf():
            return False
        else:
            return self.is_descendent(node.children[0]) or \
                self.is_descendent(node.children[1])

    def get_leaves(self):
        """Get all EDUs covered by this node.
        """
        return self._get_leaves_helper(self)

    def _get_leaves_helper(self, node):
        if node.children is None:
            return [node]
        ret = []
        for child in node.children:
            ret.extend(self._get_leaves_helper(child))
        return ret

    def promotion_set(self):
        """Return the promotion set of this node.
        """
        if self.promoset is None:
            if self.is_leaf():
                self.promoset = (self,)
            elif self.children_nuc == ("N", "N"):
                self.promoset = (self.children[0].promotion_set() +
                                 self.children[1].promotion_set())
            elif self.children_nuc[0] == "N":
                self.promoset = self.children[0].promotion_set()
            else:
                self.promoset = self.children[1].promotion_set()

        return self.promoset

    def get_marcu_score(self, root):
        """Compute a salience score from Marcu (1999).
        """
        if self._marcu_score == None:
            self._marcu_score = self._marcu_score_helper(root, root.get_height()+1)
        return self._marcu_score

    def _marcu_score_helper(self, root, height):
        if self in root.promotion_set():
            return height
        elif root.is_leaf():
            # For rare cases where root and leaf are equal but Python does
            # not think so
            return height
        elif self.is_descendent(root.children[0]):
            return self._marcu_score_helper(root.children[0], height-1)
        else:
            return self._marcu_score_helper(root.children[1], height-1)

    def __repr__(self):
        return self.print_node(self)
        
    @staticmethod
    def print_node(node, indent=2):
        """Print the current node in indented tree format.
        """
        indentstr = ''.join([' ' for i in range(indent)])
        if node.children is not None:
            return "({0}[{1}][{2}]\n{3}{4}\n{3}{5})".format(
                node.label,
                node.children_nuc[0],
                node.children_nuc[1],
                indentstr,
                RSTNode.print_node(node.children[0], indent+2),
                RSTNode.print_node(node.children[1], indent+2))
        else:
            #if node.eop:
            #    eoinfo = ' <P>'
            #elif node.eos:
            #    eoinfo = ' <s>'
            #else:
            #    eoinfo = ''
            return "{0}".format(node.label)#, eoinfo)


