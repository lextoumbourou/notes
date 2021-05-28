"""
bad_credit.py

Computer Science 50 in Python
Problem Set 1
"""

def is_valid_credit_card(card_num):
    """
    Checks if valid credit card by:
    1. Multiply every other digit by 2, starting with second-to-last digit, then add products digits together.
    2. For the ones not multiplied, add to other digits
    3. If total % 10 is 0, number is valid
    """
    first_total = second_total = 0
    for c, val in enumerate(reversed(card_num)):
        c += 1
        if c % 2 is 0:
            # Multiply value by two, then go through each character
            # and convert to an int then add to first total
            for i in str(int(val) * 2):
                first_total += int(i)
        else:
            second_total += int(val)
    
    if (first_total + second_total) % 10 is 0:
        # If the complete total is divisble by 10, we won!
        return True
    
    return False		

def is_amex(card_num):
    """Return true if first two numbers are 34 or 37"""
    first_nums = card_num[:2]
    if int(first_nums) in (34, 37):
        return True
    
    return False

def is_mastercard(card_num):
    """Return true if first two numbers are 51, 52, 53, 54 or 55"""
    first_nums = card_num[:2]
    if int(first_nums) >= 51 and int(first_nums) <= 55:
        return True
    
    return False
	
def is_visa(card_num):
    """Return true if card number starts with 4"""
    first_num = card_num[0]
    if int(first_num) is 4:
        return True
    
    return False
	
def get_input(question):
    return raw_input(question)
	
if __name__ == '__main__':
    greet = "Please type your card number: "
    while True:
        card = get_input(greet)
        try:
            int(card)
            break
        except ValueError:
            greet = "Retry: "
    
    if is_valid_credit_card(card):
        if is_amex(card):
            print "AMEX"
            exit(0)
        elif is_mastercard(card):
            print "MASTERCARD"
            exit(0)
        elif is_visa(card):
            print "VISA"
            exit(0)
                    
    print "INVALID"
    exit(1)
