# Lesson 1 - Decision Trees

* Difference between classification and regression
  * Classification
    * Take an input and map a discrete number of lables
      * Person/Animal
      * True/False
      * Male/Female
  * Regression
    * More about "continuous" value functions
      * Map a point on the x-axis to something on y (linear regression)
* Classification terms
  * Instances
    * Inputs
      * Pixels
      * Weather (like in New York Subway example)
    * "Vectors of values that define your input space"
  * Concepts
    * Function that maps inputs to outputs
  * Target Concept
    * "Thing we're trying to find"
  * Hypothesis Class
    * "Set of all concepts I'm willing to think about"
  * Sample
    * Training set
      * Set of input paired with "correct output" label.
      * Like, input data about tendancy toward violence, paired with sex
      * "Inductive learning" (as opposed to deductive learning)
* Decision Trees
  * Walk through features to help determine classification outcome
  * Candidate concept refers to the Decision Tree chosen as opposed to other choices with different features and so forth.
