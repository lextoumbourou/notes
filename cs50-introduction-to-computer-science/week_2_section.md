Running time

- time is not as important as steps

Asymptotic notation

- fancy name to look at running time

- Big O
    - worst case running time
    - most important case to look at when classifying an algorithm

- Omega
    - best case running time

- Theta
    - average case running time

- O(1) - constant (faster)
- O(log n) - logarithmic (fast)
- O(n) - linear
- O(n^2) - quadratic
- O(c^2) - exponential
- O(n!) - factorial

Linear Search
- Method
    - go through each elem in list until you get the one you want
            for n in num:
                if n is the_one_i_want:
                    return
- Big O
    O(n), Omega(1)

Binary Search
- Method (list must be sorted)
    Start in middle
    Is this right number?
            Done
    Else too high
            Divide in half
            Ignore right half
            Pear on left half
    Else too low
            Divide in half
            Ignore left half
            Repeat on right half

- Implementation
    Goal: find 9 in list
    1 3 5 6 7 8 9 11
    1. Start in middle at 6.
    2. Is this right? No. Too low.
    3. Ignore left half. 
    7 8 9 11
    4. Start in the middle at 8.
    5. Is this right? No. Too low. And so forth
	
- Big O
    O(log n), Omega(1)

Recursion
    - Recursive functions are functions that call themselves
    - They must have two parts:
        - Base call (ending condition so doesn't loop forever)
        - Recursive call
