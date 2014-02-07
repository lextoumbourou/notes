arr = [5, 100, 3, 2, 101, 100]
swap_count = 1

while swap_count:
    swap_coun = 0
    for n in range(len(arr)):
        next = n + 1
        try:
            if arr[n] > arr[next]:
                arr[n], arr[next] = arr[next], arr[n]
                swap_count = 1
        except IndexError:
            # If it's the last value, there's nothing to swap
            pass

print arr
