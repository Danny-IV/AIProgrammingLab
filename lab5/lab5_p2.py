class BinaryTree:
    def __init__(self, root_e):
        self.nodes = [root_e]

    def insert(self, e, parent, pos):
        pass

    def isLeaf(self, e):
        pass

    def delete(self, e):
        pass

    def editNode(self, e_old, e_new):
        pass

    def numOfChild(self, e):
        pass

    def isFull(self):
        pass

    def height(self):
        """Height of the tree"""
        h = 1
        while True:
            if len(self.nodes) <= 2**h - 1:
                break
            h += 1
        return h


    def __str__(self):
        """String representation"""
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
                lt[i] = '\u00B7'

        ret = lt[0]
        for d in range(1, h):
            ret += '\n' + ('\u00B7'*(2**(h-d)-1)).join(lt[2**d-1:2**(d+1)-1])
        return ret


    def __remove_trailing_none(self):
        """Remove all trailing None elements from list"""
        while True:
            if self.nodes[-1] is not None:
                break
            del self.nodes[-1]

