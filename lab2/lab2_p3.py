def resetValues(L, t):
    """returns only elements with numbers lower than t among list elements."""
    Result = []
    for num in L:
        # Only numbers less than t
        if num <= t:
            Result.append(num)  # append numbers to result
    return Result
