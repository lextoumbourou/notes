import sys

def mapper():
    """
    For each station, print the station (unit) followed by the ENTRIESn_hourly dataset for the row,
    we'll test the validity of the data, by ensuring that the row has the correct number of columns
    """
    for line in sys.stdin:
        data = line.split(',')
        # Ensure it's not the header row
        if data[2] == 'DATEn': continue
        # Ensure row has correct number of fields
        if len(data) == 21: continue
        unit, entries = data[1], data[6]
        print "{}\t{}".format(unit, entries) 

if __name__ == '__main__':
    mapper()
