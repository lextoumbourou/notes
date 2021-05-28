# Taking Derivatives

## Using secant line slopes to approximate tangent slope

* Slope
    * rate of change of a line 
    * or, rate of change of ``y`` with respect to ``x``
    * often denoted by letter ``m``
    
    <img src="./images/slope-formula.png"></img>  

* Secant line:
    * Line that intersects a curve in exactly two places

    <img src="./images/secant-line-1.png"></img>  

* Tangent line:
    * A line that touches a curve at a point without crossing over.

    <img src="./images/tangent-line.png"></img>

## Derivative as slope of a tangent line

* With a curve, the slope is different at every point.
* Firstly, understanding that you need to determine the slope a secant line between two points:
```
> change_in_y = (y[1] - y[0])
> change_in_x = (x[1] - x[0])
> slope_of_tangent_line = change_in_y / change_in_x
```
* Then, to find the slope of the line right at a point, you need to determine the limit as change approaches 0:
```
# as h approaches 0
> change_in_y = f(x + h) - f(x)
> change_in_x = x - x + h = h
```
* Examples:

<img src="./images/tangent-slope-limiting-value.png"></img>

```
>>> from math import pi, cos
>>> g = lambda x: cos(x)
>>> # value of expression when limit of h approaches 0
>>> h = -0.001
>>> (g(pi + h) - g(pi)) / h
>>> -0.0004999999583255033
>>> h = 0.001
>>> (g(pi + h) - g(pi)) / h
>>> 0.0004999999583255033
>>> # so limit appears to be 0
```

```
>>> f = lambda x: (2./3) * x - 2
>>> h = -0.001
>>> (f(1 + h) - f(1)) / h
>>> 0.66666667
>>> h = 0.001
>>> (f(1 + h) - f(1)) / h
>>> 0.66666667
>>> # Answer appears to be 2/3
```
