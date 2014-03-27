Title: Intro to Data Science Udacity Review (2014)
Tagline: 
Slug: intro-to-data-science-udacity-review
Tags: MOOC, Udacity 
Date: 2014-03-24

<div class="intro">
I thought I understood the material until I tried to use it. T-tests? A thing you do to compare two samples. Linear regression? A thing you do to try to put a line through data. But when it came down to it, I really couldn't see how it would work with real numbers.

So, I left the final project to rot for a while, while I got on living: beer, 2048, True Detective and generally making the least of life.

Lesson 3 had made me regret leaving high school. It started innocently enough but by the time this formula was scralled onto the transparent table top, I was lost at sea.

I arranged to meet with my coach the following Saturday. The lines were busy or something because I never got the call.

I reread the forums. Somebody said I should rewatch the Khan Academy videos. I did but I got lost in one of the bits. I followed the prerequisites back a few steps, to Calculs. Some of it made sense but I was missing some concepts in precalculas.

I watched some earlier Khan Academy videos. Holy fuck, it is literally taught in I traced the path back to Year 7 math.

I tried to remember the lesson it was taught in but my memory doesn't go back that far.

Days passed...
</div>

***

## Udacity so far

* Talk about the courses I've done. What I like about it, what it means to me.

## About the Course

Intro to Data Science continues along the Data Science track boasting "...?". There are now two choices, the free option, which allows you to watch the course videos and complete the prograaming questions, and the paid option. Which, for $150 a month, you get access to a "Learning Coach" and verified certificate. I paid the money with no real intention of using the coach.

The prerequistes state that you'll need a programming background to that of their CS101 program: Computer Sciense and some stats comparable to Statistics 101. I had done the statistics course a year ago or so, but it wasn't exactly fresh on my mind

## Course Overview

### Lesson 1

"Discuss data science at a high-level". Venn diagram covers definition of a data scientist, apparently, it's a "Danger Zone" if you have "substantive expertise" and "hacking skills". I'm guessing I'm hovering around there. Talks about what data scientists do.

Pi-Chaun Chang talks about work at Google, I really enjoyed this interview and felt it fit in pretty well. Gabor Szabo talks about how a data scientists takes data and finds meaning in it.

What does it mean for a data scientist to have 'substantive expertise' was apparently a recap, but I watched the first 8 videos and couldn't see where this was mentioned. However, breaking the words apart, I guess what they're getting at is it's important to have "real life" experience in the fields you're trying to analyse.

A discussion around real world problems covered some examples of problems that you'd be expected to solve as a data scientist and some you might not consider.

Creating a dataframe is covered. Some simple mapping and filtering is covered.

The class project is introduced:
  * get publicly available data on the MTA ridership
  * analyse it like a data scientist

Some advice from two data scienctists. I particularly liked Pi Chuan's advice: "think about what kind of data you're interested in and start with that". And Gabor's: "you should have a curious mind".

### Project 1

The first two problems in project set one refer to a pretty introducing problem and, in my opinion, provide a nice gentle introduction to DS. Though the 3rd question, for me, was a lot more difficult and required a reasonable amount of guess work to arrive at the right answer. The question didn't appear to be covered in the material.

### Lesson 2

Data scientists might spend "70% of their time spent data wrangling".

### Lesson 3

This lesson was really the core of the course and was important to take close notes. They advised that it would be a good idea to take stats 95 before starting the class. I had done it the year before but resat it to attempt to fill in the gaps.

A discussion around statistical significance, including how to assess feasilibity of results of a sample survey.

The discussion was quickly broken up with an interview from Kurt ??, talking about his career path. Though, I enjoyed the interview, I felt like it was inappropriately placed. It broke the flow of the lessson a little bit. I liked the intent though, to "whet ones appetite, as it were". The question "why is statistics important in data science" was answered as "to make sure you're making reasonable inferences from data".

Statistical tests: many use an assumption about the distribution of your data. Most common: normal distribution. Some exercises around the normal distribution were covered. They didn't feel particularly useful though?

Two sample t-test comes in a few variations depending on assumptions about your data. A discussion around Welch's t-test followed. The p-value "probability of obtaining a test statistic at least as extreme as the one observed".

Performing Welch's t-test in Python was covered, and obviously this was useful for the final project.

Next, non-parametric tests: statistical tests that do not assume data is normal. Again, probably useful for final project. Very interesting to learn, though difficult to see how it would fit into real life. Mann-whitney u-test where the examples given. What's the difference between the t-test and the mann_whitney test?

The second-half focuses on a rudimentary analysis of Machine Learning. A comparison of statistics vs Machine Learning (Machine Learning is less about anaylsis and more about making preditions). Then a discussion around Supervised Learning (spam filter, cost estimate of property) and Unsupervised Learning.

Next the discussion around Regression begun using Gradient Descent. I really got stuck here - close enough to give up if it wasn't for the money getting sucked out my account on a weekly basis. In the end, I followed this path to get the prerequisites to understand the material,

In the end, I understood it as:

We take some input variables (like age, weight, height), then multiple each by some value (called Theta) and add them together and use that to come up with an "output" variable.

Coefficient of determination (r^2), covered next, allows you to determine how effective your model is. The programming exercise was fairly straight forward, just covering a mathematical formula to Python code.

Concludes with an appetite whetener for other algorithms for linear regression and some additional considerations.
