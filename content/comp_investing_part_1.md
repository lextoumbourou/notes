Title: Computational Investing, Part I (Coursera) Review
Tagline: The highs and the lows of a MOOC offering from Georgia Tech
Date: 2013-10-31
Status: draft

</p>

<div class="intro">
A couple of weeks ago I completed my first Coursera course, Computational Investing, Part I run by Tucker Balch a professor at 	Georgia Tech. I have been running a site for a couple of years now called <a href="http://magicranker.com">MagicRanker</a>. Essentially a very simple implementation of a thing called The Magic Formula from a book on investing I read. I have a decent sum of money invested in the market following this approach so I guess you could say I have some experience with "computational investing". But if you were to say that I had any idea what I was doing, you'd be wrong. So, when I saw the course was available on Coursera I leaped at the idea.

</p>

The purpose of this article is to provide a sort of "review" of my experience in both the course content and Coursera as a service.
</div>

## About the Course

It's run by a Professor at Georgia Institure of Technology called Tucker Balch. Tucker is an ex-fighter pilot (according to his opening video) who has a PhD in Robotics & Machine Learning and is the founder of a quantite analysis company called Lucena Research.

It's aimed primiarly at beginners to the world of Computational Investing but who have a bit of experience programming. 

## Signature Track

Tucker's class was one of the first Coursera courses to utilise the [Signature Track](http://blog.coursera.org/post/40080531667/signaturetrack). Which, for a $50 fee, provides identity verification through ID scans and "biometric profiling" and, therefore, is able to give you a certificate with a little more credibility. I decided to sign up for the Signature Track program and I'm glad I did. Like a lot of people, I imagine, one of the challenges in MOOCs is self-motivation, without any sort of incentive, other than an unverifiable certificate at the end, it's easy to loose interest in the process and add completing the course to the "to do sometime later" pile.

By being slightly out of pocket, I definitely felt a motivation to finish and it felt over all a little more "official". It's difficult to tell your friends that you can't drink with them because you've got homework due when you're completeing a self-paced, free online course, but <span class="pull_quote right">by being a bit out of pocket through Signature Track, it was easier to pass on social events in favour of the course.</span>

## Prequistes

The prerequites fit my description pretty closely: "intended for folks who have a strong programming background, but who are new to finance and investing".

I went into this course with absolute *zero* financial experience. I have read a couple of books on value investing: [Peter Lynch](http://www.amazon.com/Beating-Street-Peter-Lynch/dp/0671891634), [Ben Graham](http://en.wikipedia.org/wiki/The_Intelligent_Investor) and, from the last paragraph, The Magic Ranker. So, I'm far from an expert but I guess I have a slight penchant toward it.

I have spent the last couple of years coding mostly in Python, and I feel pretty comfortable with it. I've got most of the Python internals down pat but hadn't played with many of the data science tools like NumPy and Pandas much. I'm also fairly comfortable with intermediary Computer Science topics, though, most grad-level Comp Sci majors would run rings around me.

I also have very little formal education. My education has mostly been on-the-job and I currently work as a Senior Engineer. I have completed a number of MOOCs and half-completed a number more. <span class="pull-right">Despite the persistant critisims, the MOOC phenomena over the last couple of years has honestly improved almost every part of my life and I'm indebt to the people who are willing to take the time to change us for the better.</span>

## Course Overview

### Week 1

The first week begins with an overview of the course. It was mentioned in one of the first slides that "student's will grade each other". I'd heard about this in Coursera and was interested to see how well it would work. However, all assigments turned out to be server graded. Given some extremely busy work circumstances at the time, I was thankful for that.

The next modules touched on the basics of hedge fund management, including how they're paid and how they attract investors. Then, the last lot of modules touched on what to expect from the rest of the course.

### Week 2

The sophomore week, covered methods of estimating company value using balance sheet metrics, news and other information. The concept of an event study is introduced.

In the second module, the "Capital Assets Pricing Module" is introduced and the implication of it for investors. If I'm honest, I found the presentation of the topic a little dry and hard to follow. [MBA Bullshit]() had a far better [CAPM Introductory series](http://www.youtube.com/watch?v=LWsEJYPSw0k). I think it's important to stress that it wasn't necessarily through any fault of Tucker's that I was bored; it seems like this topic is not all that interesting to me.

### Week 3

NumPy is introduced in the first couple of modules. That shit was a real eye-opener for me. Learning NumPy and, eventually, pandas was probably the most useful part of the course for me. 

The video tutorials covering NumPy were, however, a little slow for my tastes and, as they were mostly just a video version of tutorials available on the course's [wiki](), I ended up preferring the text-only versions of the tuts.

The course also comes with a quant library called [QSTK]() which I've found to be extremely useful. I really have to reiterate how thankful I am to Tucker and the course team for making the library available online.

### Week 4

Market arbitrage, essentially the various ways quant investors find discrepancies between price and value, is introduced. The Effective Market Hypothesis is introduced.

The sound quality for most of the modules in this week was particularly poor. Clearly Tucker and the team were experimenting with video recording. I'm not one to complain, of course :)

Lastly, a lengthy discussion around Portfolio Optimisation is covered and the first homework assigment is introduced: creating a "brute force" optimiser. I found the course work to be fairly easy as most of the code was already provided. I'm not one to complain, of course :)

### Week 5

Another extremely useful week for me. Bolliger Bands are introduced, a discussion around the different between "closing" price and "actual" price, where we learn that "actual close" represents closing price with things like stock splits and dividends payments are factored in.

It also covers practical techniques for dealing with data like techniques for filling in missing data values and for performing data sanitization checks.

The homework here was a bit more difficult creating a rudimentary market simulator and event study. I found that the amount of work required to pass the exam was relatively small but to actually complete all the assigments (which were ungraded) was a reasonable amount more work. Again, due  to some shitty work circumstances, I was thankful for this.

### Week 6

Information about how to assess event studies is covered next. Then, some different investing strategies are compared: Warren Buffett vs Jim Simons is used as the case studies.

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

## What I Would Live to See Improved

The course material was a little dry and, due to my Gen Y attention span, I found the lectures a little long. I think 5 mins is the max each lecture needs to be. I really like the way Udacity adds questions along the way to test your knowledge of each part. The course started with a couple of them but by the end, they seized.

### Coursera Interface

Yick. Not a big fan. I hated how the video couldn't be made full screen. But, I worked out how to download them to my local machine and found it preferable to using the interface

### Forums - piazza

I find it a little disorientating. Especially coming from Udacity's forum system where posts can be linked to modules. I didn't spend enough time on it. I did start getting the hang of it toward the end though and next course I'll probably do a better job of posting on it.

## Summary

Tucker was saying that the content will be revamped for the next course. When it is and if you've got an interest in finance and a reasonable programming background, you will be laughing.
