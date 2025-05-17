def SmoothingMethod(I, k):
    """return smoothed list
    input of array I, window size k"""
    expandSize = (k - 1) // 2

    # expand list with edge values
    list = [I[0]] * expandSize + I + [I[-1]] * expandSize
    smoothed_list = []

    # calculate average at list
    for i in range(len(I)):
        values = list[i : i + k]
        average = round(sum(values) / k, 1)
        smoothed_list.append(average)

    return smoothed_list
