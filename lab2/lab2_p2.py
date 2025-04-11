num = int(input("Up to which number to print?: "))

# setting row and column number
row = 0
if num < 10:
    row = num
else:
    row = 10

col = num // 10 + 1

# print numbers
for i in range(1, row + 1):
    for j in range(col):
        n = j * 10 + i
        # print n which is under "num"
        if n <= num:
            print("{:3}".format(n), end="")
    print()
