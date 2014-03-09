# Linear Regression (from Intro to Statistics)

* Lines
    * Function of y usually specified in following form:
        * ```y = bx + a```
    * Work out coefficients by first calculating a. a is the value when x == 0
* Linear Regression
    * Given data
    * Get (a, b)
    * Find line that minimizes distance in y-direction

    <img src="./images/linear-regression.png"></img>

* Regression formula (used to determine b)
    * formula

    <img src="./images/regression-formula.png"></img>

    * in code:
    ```
    def regression(x, y):
        mean_x = mean(x)
        mean_y = mean(y)
        numerator = 0
        denominator = 0
        for i, j in zip(x, y):
            result += (i - mean_x) * (j - mean_y)
            denominator += (x - mean_x) ** 2

        return numerator / (denominator)
    ```
* Calculating a
    * ```a = mean_y - b * mean_x```
