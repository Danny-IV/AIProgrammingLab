H = int(input("Display height?: "))
N = int(input("Up to which number to print?: "))

arr = [[] for _ in range(H)]

# initialize parameters
n = 1
i = 0
j = 0
state = "right"

# iterating by states
while n <= N:
    arr[j].append(n)
    n += 1

    if state == "right":
        i += 1
        state = "bottom-left"

    elif state == "bottom-left":
        i -= 1
        j += 1
        # if wall-hit change direction
        if i < 0 or j >= H:
            i += 1
            j -= 1
            if j + 1 < H:  # bottom
                j += 1
            else:  # right
                i += 1
            state = "upper-right"

    elif state == "upper-right":
        i += 1
        j -= 1
        # if wall-hit change direction
        if j < 0:
            j += 1
            state = "bottom-left"

for row in arr:
    for item in row:
        print(f"{item:3}", end="")
    print()
