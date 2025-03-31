n1 = int(input("Input integer 1: "))
n2 = int(input("Input integer 2: "))
a, b = n1, n2

if n2 > n1:
    n1, n2 = n2, n1

result = n1 * n2

while n2 > 0:
    n1, n2 = n2, n1 % n2

result //= n1

print(
    "The least common multiple of " + str(a) + " and " + str(b) + " is " + str(result)
)
