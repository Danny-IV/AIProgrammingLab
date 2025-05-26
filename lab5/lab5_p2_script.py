from lab5_p2 import BinaryTree

bt = BinaryTree("a")

bt.insert("b", "a", "l")
bt.insert("c", "a", "r")
bt.insert("b", "b", "l")  # Name already used!
bt.insert("k", "x", "l")  # No such parent node!
bt.insert("d", "a", "l")  # Position colliding with 'b'!
bt.insert("d", "a", "r")  # Position colliding with 'c'!
bt.insert("b", "a", "r")  # Duplicate name will be detected first
print(bt, end="\n\n")

flag_a = bt.isLeaf("a")
flag_b = bt.isLeaf("b")
flag_c = bt.isLeaf("c")
print(flag_a, flag_b, flag_c)
flag_e = bt.isLeaf("e")
print(flag_e is None)
print(bt, end="\n\n")

bt.insert("d", "b", "r")
bt.insert("e", "c", "l")
bt.insert("f", "c", "r")
bt.delete("a")
bt.delete("A")
bt.delete("d")
print(bt, end="\n\n")

bt.editNode("a", "A")
bt.editNode("q", "x")
bt.editNode("b", "e")
bt.editNode("q", "e")
print(bt, end="\n\n")

childn_of_A = bt.numOfChild('A')
print(childn_of_A)
bt.insert('g', 'e', 'l')
childn_of_e = bt.numOfChild('e')
print(childn_of_e)
childn_of_b = bt.numOfChild('b')
print(childn_of_b)
childn_of_x = bt.numOfChild('x')
print(childn_of_x is None)
print(bt, end="\n\n")

is_full_before = bt.isFull()
print(is_full_before)
bt.delete('g')
is_full_after = bt.isFull()
print(is_full_after)
print(bt, end="\n\n")

height_before = bt.height()
print(height_before)
bt.insert('h', 'e', 'r')
height_after = bt.height()
print(height_after)

print(bt)