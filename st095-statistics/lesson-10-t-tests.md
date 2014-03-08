# Lesson 10: T-Tests

* Determine the sample standard deviation using Bessel's Correction #'
    * S = sqrt( variance / (n - 1) )
* t-distribution
    * more prone to error
    * more spreadout
    * the larger n is (the sample size)
        * the closer the t-dist is to normal
        * the tails get skinnier
        * the closer S is to the population standard deviation (sigma)
* z-test works when we know ```mu``` and ```sigma```, but what if we don't have it? Only have samples.
* t-distribution
    * more prone to error than normal distribution
* t-statistic is sometimes called student's
* Degrees of Freedom
    * Example: if you have 3 marbles to put in 3 cups
        * 1st cup: 3 choices of marbles
        * 2nd cup: 2 choices of marbles
        * 3rd cup: 1 choice
        * Therefore, the last cup is forced, so you have *2* degrees of freedom
* t-table
    * tells the critical values in the body
    * on the left, degrees of freedom
* Finch example (birds)
    * Scientists map a trait of the birds like beak width
    * Average beak width = 6.07mm
    * Do Finches today have different-sized beak widths than before?
    * Null = beak width == 6.07mm 
    * Alternate = beak width != 6.08mm
    * Sample size = 500, df = 499
    * x-bar = average_of_sample = 6.4696
    * Std dev = sqrt(variance(sample)) = 0.4
    * t-statistic = ```(x-bar - mu) / (Std_dev / sqrt(n))``` = 22.36
    * We can definitely reject the null

    <img src="./images/finch_t_statistic.png"></img>
* p-value
    * one-tailed =  p-value probability above the t-statistic
    * two-tailed test
        * p-value probability above the t-statistic
        * p-value probability below the negative version of t-statistic
    * find p-value interval estimate using t-table
