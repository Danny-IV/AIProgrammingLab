class BinaryTree:
    """Class that represents a BinaryTree
    BinaryTree node can have two child maximum"""

    def __init__(self, root_e):
        """initialize binarytree, root_e is root node"""
        self.nodes = [root_e]

    def insert(self, e, parent, pos):
        """insert e at parent node, with l/r position"""
        # duplicate checking
        if e in self.nodes:
            print(f"Insertion refused: Node '{e}' already exists")
            return
        # parent node existence checking
        if parent not in self.nodes:
            print(f"Insertion refused: Node '{parent}' does not exist")
            return

        parent_index = self.nodes.index(parent)
        if pos == "l":
            child_index = 2 * parent_index + 1
            side = "Left"
        elif pos == "r":
            child_index = 2 * parent_index + 2
            side = "Right"
        else:
            return  # error

        # Append list if needed
        while len(self.nodes) <= child_index:
            self.nodes.append(None)

        if self.nodes[child_index] == None:
            self.nodes[child_index] = e
        else:
            print(
                f"Insertion refused: {side} child of node '{parent}' already occupied by '{self.nodes[child_index]}'"
            )

    def isLeaf(self, e):
        """Returns true if there is no child nodes"""
        # Check there is such node
        if e not in self.nodes:
            print(f"Error: Node '{e}' does not exist")
            return

        parent_index = self.nodes.index(e)
        left_index = 2 * parent_index + 1
        right_index = 2 * parent_index + 2
        if left_index < len(self.nodes):
            left = self.nodes[2 * parent_index + 1]
        else:
            left = None
        if right_index < len(self.nodes):
            right = self.nodes[2 * parent_index + 2]
        else:
            right = None
        # If left and right both none, it is leaf
        return left == None and right == None

    def delete(self, e):
        """Delete node if it exists and is a leaf node"""
        if e not in self.nodes:
            print(f"Deletion refused: Node '{e}' does not exist")
            return
        if not self.isLeaf(e):
            print(f"Deletion refused: Node '{e}' is not a leaf node")
            return
        # delete node by making it none
        index = self.nodes.index(e)
        self.nodes[index] = None

        # delete list that not needed
        self.__remove_trailing_none()

    def editNode(self, e_old, e_new):
        """Edit e_old node to e_new
        only if there is e_old and there is no e_new"""
        if e_old not in self.nodes:
            print(f"Edit refused: Node '{e_old}' does not exist")
            return
        if e_new in self.nodes:
            print(f"Edit refused: Node '{e_new}' already exists")
            return
        # change node
        self.nodes[self.nodes.index(e_old)] = e_new

    def numOfChild(self, e):
        """Returns the number of child nodes of given node e"""
        if e not in self.nodes:
            print(f"Error: Node '{e}' does not exist")
            return
        parent_index = self.nodes.index(e)
        left_index = 2 * parent_index + 1
        right_index = 2 * parent_index + 2
        cnt = 0
        if left_index < len(self.nodes):
            cnt += 0 if self.nodes[left_index] == None else 1
        if right_index < len(self.nodes):
            cnt += 0 if self.nodes[right_index] == None else 1
        return cnt

    def isFull(self):
        """Returns whether the tree is full binary tree or not"""
        flag = True
        for node in self.nodes:
            if self.numOfChild(node) == 1:
                flag = False
        return flag

    def height(self):
        """Returns height of the tree"""
        h = 1
        while True:
            if len(self.nodes) <= 2**h - 1:
                break
            h += 1
        return h

    def __str__(self):
        """String representation, show tree when called"""
        h = self.height()
        lt = self.nodes[:]

        # Remove trailing empty nodes
        while True:
            if lt[-1] is not None:
                break
            del lt[-1]

        # Replace empty node representation
        for i in range(len(lt)):
            if self.nodes[i] is None:
                lt[i] = "\u00b7"

        ret = lt[0]
        for d in range(1, h):
            ret += "\n" + ("\u00b7" * (2 ** (h - d) - 1)).join(
                lt[2**d - 1 : 2 ** (d + 1) - 1]
            )
        return ret

    def __remove_trailing_none(self):
        """Remove all trailing None elements from list"""
        while True:
            if self.nodes[-1] is not None:
                break
            del self.nodes[-1]
