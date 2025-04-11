size = int(input("Enter a size: "))
direction = input("Enter a direction: ")

# case divided by if-else
# calculated star count by a size and i
if direction == "u":
    for i in range(size):
        print(" " * (size - i - 1), end="")
        print("*" * (2 * i + 1))

elif direction == "l":
    for i in range(size):
        print(" " * (size - i - 1), end="")
        print("*" * (i + 1))
    for i in range(1, size):
        print(" " * i, end="")
        print("*" * (size - i))

elif direction == "r":
    for i in range(size):
        print("*" * (i + 1))
    for i in range(1, size):
        print("*" * (size - i))

elif direction == "d":
    for i in range(size):
        print(" " * i, end="")
        print("*" * (2 * (size - i) - 1))

else:
    print("Not proper input!")
