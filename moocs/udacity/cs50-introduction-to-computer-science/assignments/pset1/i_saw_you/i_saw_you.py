""" 
i_saw_you.py

Computer Science 50 in Python (Hacker Edition)
Problem Set 1

Run without arguments as follows:
python i_saw_you.py 

Prompts the user for 4 different sets of input and graphs it
on the command line
"""

OPTIONS = (('M4F', 'M spotting F'),
           ('F4M', 'F spotting M'),
           ('F4F', 'F spotting F'),
           ('M4M', 'M spotting M'))

def check_input(input):
    """Return input if value"""

    # Ensure it's a valid number
    try:
        int(input)
    except ValueError:
        print "That's not a number."
        return False

    if input < 0:
        print "Can't be negative"
        return False

    return int(input)

class Graph():
    """Converts 4 numbers into a graph 20 squares high"""
    def __init__(self, numbers):
        self.numbers = numbers
        self.grid_size = 20
        self.scaled_nums = self._scale_numbers()

    def _scale_numbers(self):
        """
        Add values together, then determine percentage of each one
        return a list of percentages
        """
        output = []
        # Add numbers together 
        total = sum(self.numbers)

        for num in self.numbers:
            if num:
                # Get the individual nums percentage of total
                perc = (num/float(total))*100
                # Work out how much that percentage of 20 is
                grid_perc = self.grid_size*(perc/100)
                # Convert to int and append
                output.append(int(grid_perc))
            else:
                # Don't bother running if number is 0 
                output.append(num)

        return output 

    def display(self):
        """Return a string representing a bar chart of the numbers"""
        output = "\n\n -- Your graph results -- \n"
        for row in reversed(range(1, 21)):
            for num in self.scaled_nums:
                """Go through each value, if it's more than or equal to row value
                then we'll add to the graph, otherwise, we'll leave it blank"""
                if num >= row:
                    output += "### "
                else:
                    output += "    "
            output += "\n"

        for opt in OPTIONS:
            output += "{0} ".format(opt[0])

        return output

def get_input(option):
    """Get input from the user"""
    return raw_input(option+"? ")

if __name__ == '__main__':
    results = []
    for opt in OPTIONS:
        input = False
        while input is False:
            """Continue to loop until input is valid"""
            input = check_input(get_input(opt[1]))

        results.append(input)

    graph = Graph(results)
    print graph.display()
