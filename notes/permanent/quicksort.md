---
title: Quicksort
date: 2024-02-04 00:00
modified: 2024-02-04 00:00
status: draft
---

**QuickSort** is a [Divide-and-Conquer](divide-and-conquer.md) sorting algorithm, based on using a pivot to partition an array into 2 sub-arrays, which are sorted recursively.

Pick a pivot, usually the centre: `floor((left + right) / 2)`

Smaller items are moved to right, and larger to left of pivot.

Then run QuickSort on the 2 smaller vectors recursively until we hit the base case: a sort on a one element vector.

Diagram:

 ![](../../../_media/week-17-sorting-data-ii-part-1-quicksort.png)

Don't need to create new vector for the move operations, as we use swap operation.

The partition operation does the bulk of the work in Quick sort. This method of partition is called [[Hoare Partition]] named after inventor of Quicksort, Tony Hoare.

```
function Partition(vector, i, j)
    m <- floor(LENGTH(vector/2))
    pivot <- vector[m]
    final <- m
    while i < j do
        while vector[i] < pivot do
            i <- i + 1
        end while
        
        while vector[j] > pivot do
            j <- j - 1
        end while
        
        if i < j then
            SWAP(vector, i, j)
            if i == final then
                final <- j
                i <- i + 1
            else if j == final then
                final <- i
                j <- j - 1
            else
                i <- i + 1
                j <- j - 1
            end if
        end if
    end while
    return final
end function
```
