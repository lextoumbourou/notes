import sys

def reducer():
    """
    Read each line from std in and sum the daily ridership (second field)
    until the key is different from the current key. When that happens,
    divide the summed results by the number of days in the month to get the average. Then, push into a list of tuples.
    
    On completion, we sort the tuple list and return the top 10 values
    """
    
    entries = 0
    old_key = None
    final_results = []

    for line in sys.stdin:
        unit, hourly = line.split('\t')
        if old_key and old_key != unit:
            final_results.append(
                (old_key, entries)
            )
            entries = 0
        old_key = unit
        entries += float(hourly)
    final_results.append(
        (old_key, entries)
    )

    print(sorted(final_results, key=lambda x: x[1]))[-10:]

if __name__ == '__main__':
    reducer()
