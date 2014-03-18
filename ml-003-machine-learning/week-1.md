## Introduction

### Welcome

* Grew out of AI
    * New capabilities for computers
* Modern examples
    * Database mining
        * Dealing with large datasets
    * Stuff you can't program by hand
        * Autonomous helicopters
        * Handwriting recognition
        * Natural Language Processing (NLP)
    * Self-customizing programs
        * Amazon, Netflix product recommendations
    * Understanding human learning (brain, real AI
* Machine Learning is at the top of 12 most desired skills for IT employees

### What Is Machine Learning

* "Field of study that gives computers the ability to learn without being explicitly programmed" -- Arthur Samuel (1959)
* Well-posed Learning Problem: "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."
    * Example of T: classifing emails as spam or not spam
    * Example of E: watching you label emails as spam or not spam
    * Example of P: The number (or fraction) of emails correctly classified as spam / not spam
* Algorithms
    * Supervised learning
    * Unsupervised learning
    * Reinforcement learning
    * Recommender systems

### Supervised Learning

* Most common type of ML algorithm
* Give algorithm dataset with a set of answers and asked it to give you "more" right answer
* Regression problem
    * Continous valued output (like price)
    * Examples
        * Determining housing prices based on historical prices
            * Learning algorithm could "put a straight line through data" to do predictions

                <img src="./images/straight-line-prediction.png"></img>

            * Or, could use a quadratic function or second-order polynomial to get closer to true prediction
                * ```a*x**2 + b*x + c``` is a 2nd-order polynomial because the highest exponent of x is 2.
                * The graph of a quadratic function is a parabola (the u-shaped graph)
* Classification problem
    * Discrete valued output (0 or 1)
    * Examples
        * Determine whether a tumor is malignant or benign based on other tumors 
* Features
    * Input variables used to make predictions
    * Examples
        * Age / Tumor size

### Unsupervised learning

* 2nd most common type (out of two, I guess?)
* Instead of giving algorithm "right answer", we give it a dataset and say: "find some patterns"
* Looks for patterns/groups in data called "clusters"
    * Examples
        * Google News find articles and attempts to related them into clusters
* Cocktail party algorithm
    1. Record people talking with two microphones
    2. Separate voices using data from inputs
    3. Separate voices and background noise from two inputs 
    * Can be expressed in one-line of code: ```[W,s,v] = svd((repmat(sum(x. *x,1), size(x,1),1).*x)*x');```

## Linear Regression with One Variable

### Model Representation

* You have a dataset called "training set" 
* Notation:
    * **m** - number of training examples
    * **x** - "input" variable / features
    * **y** - "output" variable / "target" variable
    
    <img src="./images/model-representation.png"></img>
* Algorithm flow
```
Training Set -> Learning Algorithm
            |
            V
Size of house (x) -> hypothesis (h) -> estimated price (y)
```
* h is represented as:

<img src="./images/h-representation.png"></img>

### Cost Function

* Hypothesis: ```h_theta(x) = Theta 0 + Theta 1*x```
    * Theta 0 and Theta 1 are the parameters of the model

    <img src="./images/h-theta-of-x-examples.png"></img>
    
    * You want to pick a Theta 0 and Theta 1 so that h_theta(x) works out as close to y as possible for training examples
    * In more formal terms, you want to minimize your Squared Error

    <img src="./images/cost-function.png"></img>

        * (Multiple by ```1/2m``` to make the math "easier" (?))
        * This is called Cost Function or Squared Error

### Gradient Descent

* An algorithm for minimizing a cost function (and any other function) J
* Outline:
    1. Start with some ```Theta[0], Theta[1]``` (or any set of Thetas)
    2. Keep changing ```Theta[0], Theta[1]``` to reduce ```J(Theta[0], Theta[1])``` until we reach a minimum
* Depending on where you start (which Thetas you test?) you may end up with various different mininums (thing of the hill/valley example)
* Overview

<img src="./images/gradient-descent.png"></img>

* Mathematical notation:
    * Assignment: ```a := b``` (as opposed to Truth Assertion ```a = b```)
    * Alpha (number called "learning rate")
    
    <img src="./images/alpha.png"></img>
    
    * Derivative term (not covered yet)

    <img src="./images/derivative-term.png"></img>

* Algorithm should do a simultaneous update. Essentially, don't change ```Theta[0]``` until you've calculated new value for ```Theta[1]``` and ```Theta[0]```

### Gradient Descent Intuition

* Derivative term
    * What is the slope of the line?
* Alpha term
    * If it's too small, it may take too long to find the minimum
    * If it's too big, it might go too far ("overshoot the minimum") and could potentially never find the minimum
* Gradient descent will automatically take smaller steps over time - no need to decrease *alpha*.

## Linear Algebra Review

### Matrices and Vectors

* Matrix Elements (entries of matrix) 
    * Matrix == rectangular array of numbers (defined in square brackets)
    ```
    [ 1402 191 ]
    [ 1371 821 ]
    [ 949 1437 ]
    ```
    * Dimension of matrix == number of rows x number of columns
        * Above example is a ```3 x 2``` matrix
    * Refer to entries like ```Aij``` or ```A[1][2] == 191```
* Vector
    * Matrix with only 1 column
    ```
    [450]
    [123]
    [345]
    [335]
    ```
    * Above example == 4-dimensional vector
    * Vectors could be 1-indexed or 0-indexed
* Usually use capital letters to refer to matrices ```A, B, C```
* Usually use lower case to refer to vector ```x, y, z```

### Additional and Scalar Multiplication

* To add each number that corresponds in the matrix

<img src="./images/matrix-multiplication.png"></img>

    * You can only add two matricies that are the same size
* Scalar multiplication simply involves multiplying each entry in the matrix by a number

<img src="./images/scalar-multiplication.png"></img>

### Matrix Vector Multiplication

* Matrix vector multiplication
```
a = [1 3]
    [4 0]
    [2 1]
b = [1]
    [5]
```
    * steps:
        1. ```a[0][0] * b[0][0] + a[0][1] * b[1][0]```
        2. ```a[1][0] * b[0][0] + a[1][1] * b[1][0]```
        3. ```a[2][0] * b[0][0] + a[2][1] * b[1][0]```
    * Result will be 3 x 1 matrix (3-dimensional vector) 
    * Can be computational efficient to do it this way instead of using an iterator

### Matrix Matrix Multiplication

* Key steps to know for gradient descent
* Example
```
a = [1 3 2]
    [4 0 1]
b = [1 0]
    [0 1]
    [5 2]
```
    * Split b in half and do matrix/vector multiplication to both sides
    * 2 x 3 * 3 x 2 = first two x second two = 2 x 2 
    * 2 x 2 * 2 x 2 = first two x second two = 2 x 2
