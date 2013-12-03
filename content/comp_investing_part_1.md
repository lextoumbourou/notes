Title: Computational Investing, Part I (Coursera) Review
Tagline: The highs and the lows of a MOOC offering from Georgia Tech
Date: 2013-10-31
Status: draft

</p>

<div class="intro">
I was frantically trying to debug an ugly script. It was supposed to simulate an event study based on Bollinger band indicators, but I couldn't get the output to match the example. Mine said the Sharpe ratio was somewhere above 2, it was supposed to be 0.878.

<p>

"Just fucking do it later," Kelly yelled from the other room, "Our plane is leaving in like 2 hours."

</p>

<p>

"Okay, okay, I'm packing!"

</p>

<p>
I stuffed a couple of shirts into the bag, then sat down to scour the course's forum for help. The last couple of weeks had been exhausting marathon of work and study. I was determined to leave every part of it behind for the week. I knew I was close to the answer. I had understood the task, I was certain, but why was the computer being cruel?
</p>

<p>

Then Gene entered my life. Gene posted on the forum without an avatar. Judging by Gene's liberal use of caps, Gene had problems of his own. But he had reminded me that there should "be no BUY orders in the first and last 20 days"!. I remembered something about that in one of the videos and, honestly, I didn't understand why that was the case, but I cut 20 days off the start and end of the data sequence, punched the values into the exam and, after they verifed my identity, passed.
</p>

<p>
I heard a horn in the driveway.
</p>

<p>
"The cab is here!", she yelled.
</p>
</div>


## About the Course

Computational Investing, Part I was instructed by Tucker Balch, a professor at Georgia Institure of Technology. Tucker is an ex-fighter pilot (according to his opening video), the founder Lucena Research and has a PhD in Robotics and Machine Learning.

The course page notes that it is "intended for folks who have a strong programming background, but who are new to finance and investing". I would say that's a reasonable fit.

## My Background

I came across the course through someone's comment on Hacker News, I think, and signed up almost immediately.  I have been running a site for a couple of years called [MagicRanker](http://en.wikipedia.org/wiki/Efficient-market_hypothesis), a very simple implementation of a thing called the Magic Formula from a small book called [The Little Book That Beats The Market](http://en.wikipedia.org/wiki/Efficient-market_hypothesis). I have a decent amount of money invested in the market following this approach so I guess you could say I have some experience with "computational investing". But, it would be wrong of me to suggest I had any idea what I was doing in the financal world.

I have, however, spent the last couple of years coding mostly in Python, and I'm pretty comfortable with it. Though at the time I hadn't played with many of the data science libraries like NumPy and Pandas. I'm also fairly comfortable with intermediary Computer Science topics.

## Signature Track

Tucker's class was one of the first Coursera courses to utilise the [Signature Track](http://blog.coursera.org/post/40080531667/signaturetrack). Which, for a $50 (AUS) fee, provides identity verification through ID scans and "keystroke biometric" profiling (apparently which turns out to be a [thing](http://en.wikipedia.org/wiki/Efficient-market_hypothesis)!)</a> and, therefore, is able to give you a certificate with a little more credibility. I decided to sign up for it and I'm glad I did. Like a lot of people, I imagine, one of the challenges in completing MOOCs is maintaining motivation until the course end. <blockquote class="pull_quote right">By being slightly out of pocket though Signature Track, I felt a sense of commitment that helped me stay focused to the end.</blockquote> Plus, I feel like it felt more "official".

Interestingly, according to Tucker's blog, the "completion rate for MOOC students who invested $40.00 at the beginning of the course for a validated certificate was 99.0%." A good sign for MOOCs, who, journalists seem to love to bemoan for their apparently [low completion rates](http://www.timeshighereducation.co.uk/news/mooc-completion-rates-below-7/2003710.article].

## Course Overview

### Week 1...

...begins with an overview of the course. It was mentioned in one of the first slides that "students will grade each other". I'd heard about this and was interested to see how well it works. Turns out, however, that all assigments were server graded. Given some extremely busy life circumstances at the time, I was thankful.

The next lot of modules covered the basics of hedge fund management, including how managers are paid and how they attract investors. Then, a glimpse of what to expect from the rest of the course.

### Week 2...

...covered methods of estimating company value using balance sheet metrics, news and other information. Event studies are discussed.

The "Capital Assets Pricing Module" is introduced detailing the implications of it for investors. If I'm honest, I found the presentation of the topic a little dry and hard to follow. [MBA Bullshit](http://www.mbabullshit.com) had a far better [CAPM Introductory series](http://www.youtube.com/watch?v=LWsEJYPSw0k) for me.

### Week 3...

...was a joy. [NumPy](http://www.numpy.org) is introduced, as is [pandas](http://pandas.pydata.org) through a series of video tutorials based on a text-based series.

The video tutorials were a little slow for my tastes and I ended up preferring the text-only versions on the [wiki](http://wiki.quantsoftware.org/index.php?title=Numpy_Tutorial_1).

The course also covered a quant library built for the course called [QSTK](http://wiki.quantsoftware.org/index.php?title=QuantSoftware_ToolKit). Really, such an incredible amount of thought went into the course. And all for free. Crazy. 

### Week 4...

...delved deeper into hedge fund tactics like market arbitrage: the various ways investors exploit market inefficiencies. A theory called [The Effective-market Hypothesis](http://en.wikipedia.org/wiki/Efficient-market_hypothesis) is introduced.

The sound quality for most of the modules for this week was particularly poor. Though Tucker apologies and it's fixed the week after.

Lastly, a lengthy discussion around Portfolio Optimisation is covered and the first homework assigment is introduced: creating a "brute force" optimiser. I found the homework to be a reasonable magnitute easier than understanding the lectures - certainly wasn't complaining.

### Week 5...

...was another extremely useful week for me. Bolliger Bands were introduced, then a discussion around the different between "closing" price and "actual" price (actual close factors things like stock splits and dividends).

Additionaly modules also covered practical techniques for dealing with data like techniques for filling in missing data values and for performing data sanitization checks.

The homework here was a tad more difficult then last week's, creating a rudimentary market simulator and event study. But, with a bit of effort I was able to piece something together and passed fairly quickly.

### Week 6

...had information on how to assess event studies. Then, some different investing strategies are compared: Warren Buffett vs Jim Simons is used as the case studies.

An indepth discussion about CAPM is next. Again, I had trouble following this, partly, I'm sure, to do with a lack of interest.

### Week 7

From the financial perspective, probably the most practical Week. Talks about different Information Feeds available for hedge fund managers.

Most of this week however was set aside to talk about the homework assigment, an implementation of Bolliger Bands. Again, I found the amount of work required to actually pass the exam was a reasonable order of magnitude less than required to complete all the assigments.

### Week 8

The final module. Centered around two homework assignments building from previous   week's work. Where, in the first, we were to perform an event study based on Bollinger bands and, in the second, feed that data into a market simulator.

I wish I had had more time to spend on them but ended up knocking together a half baked solution in order to pass the exam the night before I left for a weeklong holiday with my girlfriend. I'm not proud of the code, and I doubt I would have done very well if an instructor graded it, but I got it done and passed. This also gave me some ideas for ways to improve MagicRanker (coming soon).

The Week concludes by covering Jensen's Alpha, a measure of evaluting mutual fund managers. I had difficulty following the lesson and also limited time. Then, elaborates on back testing, including the risks, components and considerations (a very useful video) and a small module on Machine Learning.

## What I Loved About It

Numpy. Pandas. Holy.

Two of the most incredible libraries I have ever played with. With the little bit of knowledge I got from this course and [this book](http://shop.oreilly.com/product/0636920023784.do), I was able to rewrite the internals of MagicRanker and make it a ton fast and more extensible. For that reason alone, this course was worth every second of it I took. This also has lead me to want to start experimenting a lot more.

I was also quite thankful that the homework was relatively easy. Hard enough to keep me thinking but, for a person with a decent amount of programming experience, definitely passable.

I also thoroughly enjoyed the practical side of the course, far more so than the theoretical side.

## What I Would Live to See Improved

The course material was a little dry and, due to my Gen Y attention span, I found the lectures a little long. I think 5 mins is the max each lecture needs to be. Udacity's model of providing question-based "checkpoints" along the way, really helps to keep engagement up. The course had a couple of them in earlier videos but they seemed to disappear completely by module two.

### Coursera Interface

I found it quite distracting that the video couldn't be made full screen via the UI. However, the videos are made downloadable and I found it preferable to using the interface

### Forums - piazza

I find it a little disorientating. Especially coming from Udacity's forum system where posts can be linked to modules. I didn't spend enough time on it. I did start getting the hang of it toward the end though and next course I'll probably do a better job of posting on it.

## Summary

Tucker was saying that the content will be revamped for the next course. When it is and if you've got an interest in finance and a reasonable programming background, you will almost certainly find this course worthwhile. All I can say is, thank you so much to the whole team for the opportunity and I cannot wait for Part 2!
I also have very little formal education. My education has mostly been on-the-job and I currently work as a Senior Engineer. I have completed a number of MOOCs and half-completed a number more. <span class="pull-right">Despite the persistant critisims, the MOOC phenomena over the last couple of years has honestly improved almost every part of my life and I'm indebt to the people who are willing to take the time to change us for the better.<
