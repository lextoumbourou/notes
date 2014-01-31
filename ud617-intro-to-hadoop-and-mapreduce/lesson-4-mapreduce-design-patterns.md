# Lesson 4: Mapreduce Design Patterns

* Book to read: Mapreduce Design Patterns
* Overview of Patterns
    * Filtering Patterns
        * Sampling data
        * Top-N list (top 10, for example)
    * Summarization Patterns
        * Counting
        * Min/Max
        * Statistics
        * Index
    * Structural Patterns
        * Combining data sets
* Filtering Patterns
    * Don't change summarization of data
    * Go through entire data and decide what to keep and what to throw away
    * Types
        * Simple filter
            * Function that returns keep or throwaway for each record
        * Bloom filter
            * Efficient probabilistic filter
        * Sampling
            * Make a smaller dataset from a larger dataset
        * Random Sampling
            * Representative sample
        * Top K problems
* Filtering Exercise
    * Forum posts
        * Posts that are one-sentence or less
    * Problem overview
        * Find posts where body is only one-sentence as defined:
            * None of the following: '.!?'
            * One of them as last character in body
* Top 10
    * Traditional method with RDBMS
        * sort data
        * pick top N records
    * MapReduce
        * Mappers generate "local" top N lists
        * Reducer finds  "global" list
    * Problem overview
        * Write the mapper that finds the top N records
* Summarization Patterns
    * Inverted Index
        * Like index in "back of the book"
    * Numerical Summarizations
        * Count
        * Min/max
        * First/Last
        * Mean, Median
* Numerical Summarizations
    * Word/record count
    * Mean/median/std dev
* Mean and Standard Deviation
    * Work out something like: Any correlation between the day of the week and how much people spend on items?
* Combiners
    * "prereduction"
