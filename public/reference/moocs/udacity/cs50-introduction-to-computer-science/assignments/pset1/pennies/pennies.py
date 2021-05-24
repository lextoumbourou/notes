"""
pennies.py

Computer Science 50 in Python (Hacker Edition)
Problem Set 1

Get num of days in month then works out how many $$ you'd have by end of the month
if you received a penny on the first day, two on second, four on third and so on
"""

def get_input(question):
    """Get the days input from the user """
    return raw_input(question + " ")

def is_valid_days(days):
    """Check if the number of days is between 28 and 31"""
    try:
        days = int(days)
    except ValueError:
        print "Not a valid integer"
        return False

    if days < 28 or days > 31:
        print "Not a valid number of days"
        return False

    return days

def is_valid_cents(cents):
    """Ensure number of cents is a valid int"""
    try:
        return int(cents)
    except ValueError:
        print "That's not a number."
        return False

class Exponenter():
    days = None
    cents = None
    dols = None

    def __init__(self, cents, days):
        """
        Takes an integer of cents and days
        then performs exponent analysis
        """
        self.cents = cents
        self.days = days
        self.final_cents = self._exponent()

    def __unicode__(self):
        """Return the string output in $00,000,000 format"""
        output = ""
        # Reverse the string using extended slice syntax
        rev_dols = "{0:.2f}".format(self.dols)[::-1]
        # If the dollars is more than 1000, add a comma every 3 digits
        if self.dols >= 1000:
            for count, char in enumerate(rev_dols):
                count += 1
                output += char

                if count <= 3:
                # Ignore first 3 characters, as they're decimal points
                    continue

                if count % 3 is 0 and len(rev_dols) is not count: 
                # For every 3 characters, that's not the last one, add a ,
                    output += ","
        else:
            output = rev_dols

        # Reverse the output and return it
        str_dols = output[::-1]

        return "${0}".format(str_dols)

    def __repr__(self):
        """Return the representation as float of dollars"""
        return "{0}".format(self.dols)

    def _exponent(self):
        # For each day, multiply the cents by an incrementer that doubles each time
        inc = 1
        for day in range(1, self.days+1):
            final_cents = self.cents*inc
            inc += inc

        # Return the number of dollars (cents/100)
        self.dols = final_cents/float(100)

        return final_cents

if __name__ == '__main__':
    days = False
    while not days:
        days = is_valid_days(get_input("Number of days in month? "))

    cents = False
    while not cents:
        cents = is_valid_cents(get_input("Numbers of cents on first day? "))

    exp = Exponenter(cents, days)
    print "{0}".format(unicode(exp))
