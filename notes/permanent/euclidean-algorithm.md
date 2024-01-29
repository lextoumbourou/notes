---
title: Euclidean Algorithm
date: 2023-10-30 00:00
modified: 2023-10-30 00:00
status: draft
---

An algorithm for finding the greatest common divisor of two integers.

The iterative algorithm represented in pseudocode like this:

```
function GreatestCommonDivisor(a, b)
    while a != b do
        if a > b then
            a = a - b
        else
            b = b - a
        end if
    end while
    
    return a
    
end function
```
