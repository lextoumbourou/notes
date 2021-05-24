s_list = [1,2,3,4,5,6,7,8]
val = 7

def binary(s_list, val):
    middle = 1
    while len(s_list) and middle:
        middle = len(s_list)/2
        if s_list[middle] is val:
            return val
        elif s_list[middle] < val:
            s_list = s_list[middle:]
        elif s_list[middle] > val:
            s_list = s_list[:middle]

    return False

print binary(s_list, val)
