def merge_sort(vals):
    """Take a list of numbers and sort them using merge sort"""
    list_length = len(vals)

    if list_length < 2:
        return vals

    middle = list_length/2

    first = merge_sort(vals[:middle])
    second = merge_sort(vals[middle:])

    return merge(first, second)

def merge(first, second):
    first_len, second_len = len(first), len(second)

    # If either are empty (i.e. we don't have a side to sort),
    # obviously don't bother with the merge
    if not first_len or not second_len:
        return first or second 

    i = j = 0
    output = []

    while i < first_len and j < second_len:
        if first[i] < second[j]:
            output.append(first[i])
            i += 1
        else:
            output.append(second[j])
            j += 1

    if i == first_len:
        output.extend(second[j:])
    else:
        output.extend(first[i:])

    return output

if __name__ == '__main__':
    print merge_sort([100, 5, 6, 7, 10, 11, 2, 1])
