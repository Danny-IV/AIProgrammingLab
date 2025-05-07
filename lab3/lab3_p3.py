def FindMaxValue(D1, D2, D3):
    """Find max value of dice stack
    dice can only stack by same number"""
    # initialize
    max_Value = -1

    # Dice stack number of cases
    # cases[i][0] = top, cases[i][1] = middle, cases[i][2] = bottom
    cases = [
        [D1, D2, D3],
        [D1, D3, D2],
        [D2, D1, D3],
        [D2, D3, D1],
        [D3, D1, D2],
        [D3, D2, D1],
    ]

    # dice base, upper pairs
    pairs = [[0, 4], [4, 0], [1, 3], [3, 1], [2, 5], [5, 2]]

    pair_cases = []
    for pair1 in pairs:
        for pair2 in pairs:
            for pair3 in pairs:
                pair_cases.append(
                    [pair1[0], pair1[1], pair2[0], pair2[1], pair3[0], pair3[1]]
                )

    # for all cases
    for c in cases:
        # for all pair cases
        for p in pair_cases:
            top_base = c[0][p[0]]
            top_upper = c[0][p[1]]
            middle_base = c[1][p[2]]
            middle_upper = c[1][p[3]]
            bottom_base = c[2][p[4]]
            bottom_upper = c[2][p[5]]

            # only adjacent face is same number
            if top_base == middle_upper and middle_base == bottom_upper:
                diceSum = (
                    sum(c[0])
                    + sum(c[1])
                    + sum(c[2])
                    - top_base
                    - middle_base
                    - middle_upper
                    - bottom_base
                    - bottom_upper
                )
                if diceSum > max_Value:
                    max_Value = diceSum

    return max_Value
