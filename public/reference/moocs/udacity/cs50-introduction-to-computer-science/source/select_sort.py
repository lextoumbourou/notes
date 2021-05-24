to_sort = [5, 30, 10, 4]

for outer_pos in range(len(to_sort)):
    lowest_pos = outer_pos
    for c in range(len(to_sort[outer_pos:])):
        # Don't work on stuff we've already checked
        inner_pos = c + outer_pos
        try:
            if to_sort[inner_pos] < to_sort[lowest_pos]:
                lowest_pos = inner_pos
        # If we're out of range, then there's nothing more to be done
        except ValueError:
            pass

    if lowest_pos is not outer_pos:
        # Switch the position around if we found a lower value
        to_sort[outer_pos], to_sort[lowest_pos] = to_sort[lowest_pos], to_sort[outer_pos]


print to_sort
