base = int(input("Enter the base integer: "))
d3 = int(input("Enter leftmost digit: "))
d2 = int(input("Enter the next digit: "))
d1 = int(input("Enter the next digit: "))
d0 = int(input("Enter the last digit: "))
print(
    "Your input is " + str(d3) + str(d2) + str(d1) + str(d0) + " in base " + str(base)
)
result = d3 * base**3 + d2 * base**2 + d1 * base**1 + d0
print("The value is " + str(result) + " in base 10")
