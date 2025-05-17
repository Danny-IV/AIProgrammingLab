def lookAndSaySequence(n, l):
    """returns the list of integers and cycle flag"""
    # initialize parameters
    list = [n]
    cycleFlag = False

    # iterating
    for _ in range(l - 1):
        previous = str(list[-1])

        digits = sorted(set(previous))
        num_string = ""

        for d in digits:
            count = previous.count(d)
            num_string += str(count) + d

        num = int(num_string)
        if num in list: # if there is such number, it will cycle
            cycleFlag = True

        list.append(num)

    return list, cycleFlag