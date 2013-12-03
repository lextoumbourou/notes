Title: Computational Investing, Part I (Coursera) Review
Tagline: The highs and the lows of a MOOC offering from Georgia Tech
Date: 2013-10-31
Status: draft

</p>

<div class="intro">
I was frantically trying to debug an ugly script. It was supposed to simulate an event study based on Bollinger band indicators, but I couldn't get the output to match the example.

<p>

"Just fucking do it later," Kelly yelled from the other room, "our plane is leaving in 2 hours!"

</p>

<p>

"Okay, okay, I'm packing!"

</p>

<p>
I stuffed a couple of shirts into the bag, then sat down to scour the course's forum for help. The last couple of weeks had been an exhausting marathon of work and study. I was determined to leave every part of it behind for the week. I knew I was close to the answer. I had understood the task, I was certain, but why was the computer being cruel?
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

Computational Investing, Part I was instructed by Tucker Balch, a professor at Georgia Institure of Technology. Tucker is an ex-fighter pilot (according to his opening video), the founder of Lucena Research and has a PhD in Robotics and Machine Learning.

The course page notes that it is "intended for folks who have a strong programming background, but who are new to finance and investing". I would say that's a good fit for me.

## My Background

I came across the course through someone's comment on Hacker News, I think, and signed up almost immediately.  I have been running a site for a couple of years called [MagicRanker](http://en.wikipedia.org/wiki/Efficient-market_hypothesis), a very simple implementation of a thing called the Magic Formula from a small book called [The Little Book That Beats The Market](http://en.wikipedia.org/wiki/Efficient-market_hypothesis). I have a decent amount of money invested in the market following this approach so I guess you could say I have some experience with "computational investing". But, it would be wrong of me to suggest I had any idea what I was doing.

I have, however, spent the last couple of years coding mostly in Python, and I'm pretty comfortable with it. Though, at the time I hadn't played with the cornerstone data science libraries like NumPy and Pandas. I'm also fairly comfortable with intermediary Computer Science topics.

## Signature Track

Tucker's class was one of the first Coursera courses to utilise the [Signature Track](http://blog.coursera.org/post/40080531667/signaturetrack). Which, for a $50 (AUS) fee, provides identity verification through ID scans and "keystroke biometric" profiling (apparently which turns out to be a [thing](http://en.wikipedia.org/wiki/Efficient-market_hypothesis)!)</a> and, therefore, is able to give you a certificate with a little more credibility. I decided to sign up for it and I'm glad I did. Like a lot of people, I imagine, one of the challenges in completing MOOCs is maintaining motivation until the course end. <blockquote class="pull_quote right">By being slightly out of pocket though Signature Track, I felt a sense of commitment that helped me stay focused to the end.</blockquote> Plus, I feel like it felt more "official".

Interestingly, according to Tucker's blog, the "completion rate for MOOC students who invested $40.00 at the beginning of the course for a validated certificate was 99.0%." A good sign for MOOCs, who, journalists seem to love to bemoan for their apparently [low completion rates](http://www.timeshighereducation.co.uk/news/mooc-completion-rates-below-7/2003710.article).

## Course Overview

### Week 1... 

... begun with an overview of the course. It was mentioned in one of the first slides that "students will grade each other". I'd heard about this before and was interested to see how well it works. Turned out, however, that all assigments were server graded. Given my extremely busy life circumstances, I was thankful.

The next lot of modules covered the basics of hedge fund management, including how managers are paid and how they attract investors. Then, a glimpse of what to expect from the rest of the course.

### Week 2...

... covered methods of estimating company value using balance sheet metrics, news and other information. Event studies are discussed.

The [Capital asset pricing module](http://en.wikipedia.org/wiki/Capital_asset_pricing_model) is introduced detailing the implications of it for investors. If I'm honest, I found the presentation of the topic a little dry and hard to follow. For me, [MBA Bullshit](http://www.mbabullshit.com) had a far simpler [CAPM Introductory series](http://www.youtube.com/watch?v=LWsEJYPSw0k).

### Week 3...

...was a joy. [NumPy](http://www.numpy.org) was introduced, as was [pandas](http://pandas.pydata.org), through a series of video tutorials based on a text-based series. The video tutorials were paced little slow and I ended up preferring the text-only versions on the [wiki](http://wiki.quantsoftware.org/index.php?title=Numpy_Tutorial_1). Later modules also covered a quant library built for the course called [QSTK](http://wiki.quantsoftware.org/index.php?title=QuantSoftware_ToolKit). Such an incredible amount of work went into the course. And all for free. Crazy.

The very first homework was assigned where we were to create a "brute force" optimiser. I found the homework to be a magnitute easier than understanding the lectures.

### Week 4...

...delved deeper into hedge fund tactics like market arbitrage: the various ways investors exploit market inefficiencies. A theory called [The Effective-market Hypothesis](http://en.wikipedia.org/wiki/Efficient-market_hypothesis) was introduced.

The video sound quality this week was particularly poor. I think the team were experimenting with different recording techniques, as it was drastically improved the week after. The videos alternated between opening to Ray Charles - What'd I Say and an 80s rock tune I've never heard before. Tucker would sometimes start the video clapping in excitment and some times awkwardly slide into his chair sipping noisily from a styrophone cup labelled "COFFEE". Kelly got a kick out of these.

Lastly, there was a lengthy discussion around portfolio optimisation and the second homework assigment was introduced: performing an Event Study. Most of the code was already provided and minor tweaks were required to pass the exam.

### Week 5...

...was another extremely useful week for me. Bolliger Bands were introduced, then a discussion around the difference between *closing* price and *actual closing* price (actual close factors in things like stock splits and dividends).

Additionally, there were modules covered practical techniques for dealing with "bad" market data from providers.

The homework here was a tad more difficult then last week's, creating a straight forward Market Simulator. But, with a bit of effort, I was able to piece something together quickly that passed. I do question how well I would have faired had a human grade my code - it was often not that pretty.

### Week 6...

...furthered the discussion around assessing event studies. Then, different investing strategies were compared using Warren Buffett and Jim Simons as opposing case studies.

An indepth discussion about CAPM followed. As before, I had trouble groking this. Partly, I'm sure, to do with a lack of interest.

The homework pieced together two of the earlier assigments, creating an event study and then running it through a simulator. With the last two weeks homework complete, the task was trivial.

### Week 7... 

..was, from a financial perspective, probably the most practical week. Videos covered the different information feeds available for hedge fund managers.

The majority of the latter half of the week was set aside to talk about the homework assigment, an implementation of Bolliger Bands. Again, I found the amount of work required to actually pass the exam was a reasonable order of magnitude less than required to complete all the assigments.

### Week 8...

... the final module, was centered around two homework assignments building from previous week's work. Where, in the first, we were to perform an event study based on Bollinger bands and, in the second, feed that data into a market simulator.

I wish I had had more time to spend on them but ended up knocking together a half baked solution in order to pass the exam the night the moment before I left for a week long holiday. I'm not proud of the code, and I doubt I would have done very well if an instructor graded it, but I got it done and passed. This also gave me some ideas for ways to improve MagicRanker (coming soon).

## What I Loved

Firstly, Tucker and the team put a lot of work into the course and it shows. Interviews, book recommendations, the QSTK, programming tutorials and a wiki rich with content.

I really got a kick out of the programming side too. Just being introduced to NumPy and Pandas made the course worth my time. With the little bit of knowledge I got from this course and [this book](http://shop.oreilly.com/product/0636920023784.do), I was able to rewrite the internals of MagicRanker and make it a shit load faster and more extendable. For that reason alone, the course was worth my time. 

I was also quite thankful that the homework was relatively easy. Hard enough to keep me thinking but, for a person with a decent amount of programming experience, definitely passable. Perhaps, though, a little too easy at times?

## What I Would Improve

The course material was sometimes a little dry and, perhaps due to my Gen Y attention span, I found the lectures a bit long. Udacity's model of providing question-based "checkpoints" along the way, really helps to keep engagement up and break up the videos. The course had a couple of them in earlier videos but they seemed to disappear completely by Week 2.

There was also a number of problems with the video quality, including low audio, distorted audio, bloopers and so forth. Sometimes I wondered why they hadn't rerecorded the videos. However, I understand how much work goes into something like this and I'm not complaining.

From the material's perspective, I had a bit of trouble with a lot of the "maths" that goes into finance. Often there's large element of plugging theoretical-world values into formulas that involve real-world results. The idea of "risk" in CAPM is a good example of this - can a standard deviation of an investments price performance really be enough to say an investment is safe or not.

Lastly, I think Coursera has some work to do with the interface. I found it slightly annoying that the video couldn't be made full screen via the UI. Not sure why they would prevent that, given they have the videos available for download. Lastly, I found the forum a little disorientating. Especially coming from Udacity's forum system where posts can be linked to modules and it's clear which video or assigment a question is about. Though, in fairness, I didn't spend enough time on it.

## Summary

Tucker was saying that the content will be revamped for the next course. When it is and if you've got an interest in finance and a reasonable programming background, you will almost certainly find this course worthwhile. If you have an interest in the world of Quant, but don't know where to start this is definitely for you. But, if you haven't programmed before, then I would definitely take an introductory course first.

Despite the persistant critisims, the MOOC phenomena over the last couple of years has honestly improved almost every part of my life and I'm indebt to the people, like Tucker and his team, who are willing to take the time to improve the lives of low-educated people like myself. We are, honestly, living in a magical time.

To conclude: thank you so much to Tucker and the team for the opportunity and I cannot wait for Computational Investing, Part 2! 
