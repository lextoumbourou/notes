Title: Intro to Data Science Udacity Review (2014)
Tagline: Another course I did, and here's why.
Slug: intro-to-data-science-udacity-review
Tags: MOOC, Udacity 
Date: 2014-03-24
Status: Draft

Summary
<div class="intro">
By Lesson 3 it got to hard so I gave up. I had regret not pursuing higher education. I let the lesson rot for a while. I caught up with important current events, like 2048, and moved on with my life

A few months later Kelly saw my credit card statement.

"Why is a company called Udacity billing you for $150 a month?"

I came back and waded through the forums for a while. A few people had said that Coursera's Machine Learning course was far better than material at Udacity.

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

I've been on Udacity for about 3 years and have had an amazing experience so far. I've learned about stuff that I never would have believed that I was capable of learning. And that's the honest truth. I've taken the introducry Computer Science course and have to say, it's one of the best out there, aside from Malen's CS50 - that's the best availalbe. The Algorithms course as well as the Programming Languages, the Programming Languages course was one of the best I've done.

## About the Course

Intro to Data Science continues along the Data Science track boasting "...?". There are now two choices, the free option, which allows you to watch the course videos and complete the prograaming questions, and the paid option. Which, for $150 a month, you get access to a "Learning Coach" and verified certificate. I paid the money with no real intention of using the coach.

The prerequistes state that you'll need a programming background to that of their CS101 program: Computer Sciense and some stats comparable to Statistics 101. I had done the statistics course a year ago or so, but it wasn't exactly fresh on my mind

## Course Overview

The course was well-planned and executed. The lessons used a narrative of analysising a baseball dataset, where the projects used the New York transit system dataset.

## Prereqs

You will definitely need:
  * a fairly strong programming background particularly in Python. You'll probably want to have played with Numpy or Pandas. If not, I would suggest spending a bit of time playing with those tools before starting this course.
  * roughly early college/uni math. I managed to get through the course with no exposure to Linear Algebra and Calculus but a fairly solid Algebra foundation. Most of the stuff is available on Khan Academy. Here's the 3 main courses I recommend you take:
    * Calculus
    * Linear Regression
    * Linear Algebra
* Statistics is also a prerequistite. I rewatched the Udacity Statistics course up to Lesson 10 and it was enough to understand most of the early half of the course.
  * Links: Statistics

## Walkthrough (skip this part if you don't intend to take the course)

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

## Lesson 4

Begins with a question: "what is information visualization?" Covers some ideas around how to do effective data visulations. Then provides a wonderful example of a clear, information visualization using [Charles Minard's flow map of Napoleon's March](http://en.wikipedia.org/wiki/File:Minard.png). Next a discussion around the "ingredients" to a good dataset.

Interviews a guy called Don Dini, he says "humans are hard-wired to receive things in story form" so one should "craft a narrative" to make the dataset more "compelling" and one should "know (their) audience". Another chap called Rishiraj Pravahan who says make it into a story so that even people who aren't familiar with the media will appreciate it.

Followed by a discussion around Visual Encodings covering the different types of visual queues one could use for representing data: position (line chart), length (bar chart) and angle (pie chart). The question around perception of visual queues was fairly unclear.

The section on Plotting in Python covered using ggplot over Matlabplot, for a number of reasons (looks nicer, has a "grammar of graphics", works well with pandas), though there were a heap of problems with ggplot due to its apparently lack of maturity.

Different data types was covered. Numeric data - any data point that has exact data (quantitative data), can be categorised into discrete (whole numbers - player's number of home runs) or continous (can fall anywhere in a range). Categorical data - position, hometown, team etc. Time series - data collected via repeated measurements over time.

An example of a bad use of scales was used. Interesting to see misleading graphs at work.

The latter half of the video walkthrough an example of charting some real-world baseball data. I enjoyed how we went step-by-step through building the visulation highlight what worked about the chart-type and what needed improvement. Starting with a scatterplot of the data, moving to a line-chart, then to a LOESS curve (a form of weighted regression), then onto multivariate data.

Rishraj's advice was: learn the tools and use them in the correct way. Don's advice was more about learning as many mathematical tools as possible.

## Lesson 5

The MapReduce lesson was pretty brief. I'd imagine for someone not familar with the content, it'd be near impossible to have a decent sense of what it does. Though it's hard to say. I've completed the Udacity Map/Reduce lesson and felt pretty comfortable with the material.

The lesson begun with an overview of where Map/Reduce is useful (huge datasets!). Talks about Map/Reduce artictecure, then covers some examples of performing Map/Reduce to perform basic word counts. Then on writing a reducer.

The example moved away using the baseball dataset to one of Aadhar data - the national Indian identification system.

Closest with a brief overview of the MapReduce ecosystem, covering mostly Pig. Then concludes with an intro to the final project.

## What was good about it.

* Pretty phenemominal course. Very diverse range of stuff covered.
