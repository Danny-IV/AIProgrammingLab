def CarSimulation(N, M):
    """Calculate rotate num(k), final column and row(x, y)"""
    # initialize variable
    k, x, y = 0, 1, 1

    # start
    total = N
    x = N
    add_n = -(N - 1)
    add_m = M - 1
    flag = False  # True(x-axis add), False(y-axis add)

    # caculating loop
    while True:
        if total >= N * M:
            break

        if flag:
            x += add_n
            total += abs(add_n)
            add_n = (abs(add_n) - 1) * (-1 if add_n > 0 else 1)
        else:
            y += add_m
            total += abs(add_m)
            add_m = (abs(add_m) - 1) * (-1 if add_m > 0 else 1)
        
        k += 1
        flag = not flag

    return k, x, y