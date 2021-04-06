def sigma(n):
    if n <= 0:
        return 0

    return n + sigma(n - 1)

print sigma(101)


