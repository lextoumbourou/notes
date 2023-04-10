---
title: Lesson 4 - Data Visualization
date: 2014-04-16 00:00
status: draft
---

# Lesson 4 - Data Visualization

* Information Visulization
    * Effective communication of complex quantitative ideas
        * Clarify
        * Precision
        * Efficiency
    * Helps you notice things about data (correlations, trends)
    * Highlight aspects of data, or "tell a story"
* Communicating findings
    * Don's advice
        1. "Craft a narrative"
        2. Know your audience
            * Technically minded?
            * People you want to recruit
            * Customers
    * Rishi's advice
        * Make is easily understandable but have math/stat rigour
* Visual Encodings
    * Position
        * Example: Positial data on chart
    * Length
        * longer the bar, greater the value
        * Example: Bar chart
    * Angle
        * Example: pie chart
            * the higher the degrees, the bigger the "slice"
            * generally avoid if showing very small differences
    * Direction
        * Has similar problems it angle. Can be hard to see differences.
    * Shape
        * use to differentiate types of data (different teams, districts)
    * Volume
        * representing data with size
    * Colour
        * hue
            * categorical data
        * saturation
            * intensity of colour for a hue
* Plotting with ```ggplot```
    1. Create plot

    ```
    > ggplot(data, aes(x_var, y_var))
    ```

        * ```data``` == dataframe
        * ```aes(x_var, y_var)```
    2. Represent data with geometic objects
        * ```geom_point()``` - change property of points
        * ```geom_line()``` - change property of lines
    3. Add labels
        *```ggtitle('Title')``` - to title plot
        * ```xlab('X Label')``` - label x
    * Example:

    ```
    >> print ggplot(df, aes(x="yearID", y="HR")) + geom_point(colour="red") + geom_line()
    ```

* Data types
    * Numeric data
        * A measurement (height, weight) or count (HR or hits)
    * Discrete and continuous
        * Discrete: Can only have whole number values
        * Continuous: any number within range
    * Categorical data
        * Represent characteristics (eg position, team, hometown, handedness)
    * Ordinal data
        * Categories with some order or ranking
        * Movie: between 1 star and 5 stars
        * Same as categorical but ordered
    * Timeseries
        * Collection of numbers collected in intervals over time
* Scale
    * Scale must be inconsistent
* Visualizing Time Series Data
    * Scatterplot without lines can make it hard to view trends
    * Linechart may focus on year-to-year variability instead of overall trends
    * LOESS curve can capture long term trends
* Multivariate data
    * Use scale to show where additional events occured with the base event
    * Double up on visual queues: area and colour
* Rishraj's advice
    * Learn tools well
    * Use them in the correct way
* Don's advice
    * Difference between good and mediocre ds
        * Feature-selection process
        * Learn as many mathematical tools as possible
