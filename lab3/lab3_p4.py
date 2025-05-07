import math


def FindGoldbachCombination(N):
    """Find Goldbach combination of N
    all combination of two prime number"""
    # Initialize
    result = []
    # Calculate prime number
    primes = []

    arr = [True] * (N + 1)
    arr[0] = arr[1] = False

    for i in range(2, int(math.sqrt(N)) + 1):
        if arr[i]:
            for j in range(i * i, N + 1, i):
                arr[j] = False
    for i in range(len(arr)):
        if arr[i]:
            primes.append(i)

    # Checking all sums
    for i in range(len(primes)):
        for j in range(i, len(primes)):
            if primes[i] + primes[j] == N:
                result.append([primes[i], primes[j]])

    return result
