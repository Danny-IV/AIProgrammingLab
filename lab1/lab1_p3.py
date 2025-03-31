pos = []
neg = []
while True:
    num = int(input("Your number: "))
    if num == 0:
        break
    if abs(num) <= 100:
        if num < 0:
            neg.append(num)
        else:
            pos.append(num)

if len(pos) == 0:
    print("No positive integer entered")
else:
    print(
        "There are "
        + str(len(pos))
        + " positive integer(s) and the sum is "
        + str(sum(pos))
    )
if len(neg) == 0:
    print("No negative integer entered")
else:
    print(
        "There are "
        + str(len(neg))
        + " negative integer(s) and the sum is "
        + str(sum(neg))
    )
