Title: Computational Investing, Part I (Coursera) Review
Tagline: "I'll give you a MOOC!"
Date: 2013-12-02

</p>

<div class="intro">
I was trying to debug an ugly script. It was supposed to simulate an event study using Bollinger Bands, but I couldn't get the output to match the example.

<p>

"Fucking finish it later," Kelly yelled from the other room, "our plane is leaving in 2 hours!"

</p>

<p>

"Okay, okay, I'm packing!"

</p>

<p>
I stuffed a couple of shirts into a bag, then sat down to search the course's forum for help. The last couple of weeks had been a marathon of work and study and I was determined to leave it all behind for the week. I knew the answer was close. I had understood the task, I was certain, but the interpreter was being stubborn.
</p>

<p>
Then Gene entered my life. Gene posted on the forum without an avatar. Judging by Gene's liberal use of caps, Gene had problems of his own. Gene reminded me, however, toward the end a rant, that there should be "no BUY orders in the first and last 20 days!". I remembered something about that in one of the videos and, honestly, I didn't understand why that was the case, but I cut 20 days off the start and end of the data sequence, punched the values into the exam and...passed!
</p>

<p>
I heard a horn in the driveway.
</p>

<p>
"The cab is here!"
</p>
</div>

</p>

* [About The Course](#about-the-course)
* [My Background](#my-background)
* [Signature Track](#signature-track)
* [Course Overview](#course-overview)
* [What I Loved](#what-i-loved)
* [What I Would Improve](#what-i-would-improve)
* [Summary](#summary)

***

<a name="about-the-course"></a>

## [About The Course](#about-the-course)

Computational Investing, Part I was instructed by Tucker Balch, a professor at Georgia Institute of Technology. Tucker is an ex-fighter pilot (according to his opening video), the founder of [Lucena Research](https://lucenaresearch.com/) and has a PhD in Robotics and Machine Learning.

The course page notes that it is "intended for folks who have a strong programming background, but who are new to finance and investing"; a close enough description to me.

<a name="my-background"></a>

## [My Background](#my-background)

I came across the course through someone's comment on Hacker News and signed up immediately.  I have been running a site for a couple of years called [MagicRanker](http://en.wikipedia.org/wiki/Efficient-market_hypothesis), a simple implementation of a thing called the Magic Formula from a book called [The Little Book That Beats The Market](http://www.amazon.com/Little-Still-Market-Books-Profits/dp/0470624159). I have money invested in the market following this approach so I guess you could say I have experience with "computational investing". But, generally I have no idea what I'm doing.

I have, however, spent the last couple of years coding mostly in Python and I'm pretty comfortable with it. So I was lucky that the course's projects were all Python based. Though, at the time though I hadn't played with the cornerstone data science libraries like NumPy and Pandas.

<a name="signature-track"></a>

## [Signature Track](#signature-track)

Tucker's class was one of the first to utilise Coursera's [Signature Track](http://blog.coursera.org/post/40080531667/signaturetrack). Which, for a $50 (AUS) fee, provides identity verification through an ID scans and "keystroke biometric" profiling (apparently a [thing](http://en.wikipedia.org/wiki/Efficient-market_hypothesis))</a> and is therefore able to give you a credible certificate. I decided to sign up for it and I'm glad I did. My problem with MOOCs is that I often find myself distracted with life halfway through the course, and emptily promise that I'll come back to it one day. <span class="pull_quote right">By being slightly out of pocket though Signature Track, I felt a sense of commitment that helped me prioritise the course.</span> Though I didn't like the idea at first, I think the fact Coursera has hard start and end dates for their courses helps with this too.

Interestingly, according to Tucker's blog, the "completion rate for MOOC students who invested ... at the beginning of the course for a validated certificate was 99.0%." A good sign for MOOCs, who journalists seem to love to bemoan for their apparently [low completion rates](http://www.timeshighereducation.co.uk/news/mooc-completion-rates-below-7/2003710.article).

<div class="img-annotated">
    <img src="/images/comp_investing_verified.jpg" alt="Signature track">
</div>

<a name="course-overview"></a>

## [Course Overview](#course-overview)

Week 1 begun with an overview of the course. It was mentioned in one of the first slides that "students will grade each other". I'd heard about this before and was interested to see if it would work. Turned out, however, that all assigments were server graded. Given my busy circumstances, I was thankful. The next modules covered the basics of hedge fund management, including how managers are paid and how they attract investors. Then a look at what to expect from the rest of the course.

Week 2 opened with a series on estimating company value using balance sheet metrics, news and other information. Event studies were also discussed, followed by a lecture on [Capital Asset Pricing Model (CAPM)](http://en.wikipedia.org/wiki/Capital_asset_pricing_model). If I'm honest, I found the presentation of this topic a little dry and hard to follow. For me, [MBA Bullshit](http://www.mbabullshit.com) had a far simpler [CAPM Introductory Series](http://www.youtube.com/watch?v=LWsEJYPSw0k).

Week 3 was a joy. [NumPy](http://www.numpy.org) was introduced, as was [pandas](http://pandas.pydata.org), through a series of video tutorials that accompanied a [text-only series](http://wiki.quantsoftware.org/index.php?title=Numpy_Tutorial_1). The video tutorials were paced a little slow for me and I generally preferred the text version. Next a quant library built for the course called [QuantSoftware ToolKit](http://wiki.quantsoftware.org/index.php?title=QuantSoftware_ToolKit) was introduced, really highlighting the richness of resources provided by the team.

The very first homework was assigned, creating a "brute force" portfolio optimiser. I found the assigment to be easier than understanding the lectures, at least the parts of the assigment required to pass.

Week 4 delved into market arbitrage: the various ways investors exploit market inefficiencies. A theory called [The Effective-market Hypothesis](http://en.wikipedia.org/wiki/Efficient-market_hypothesis) was covered. The video sound quality this week was particularly poor. I think the team was experimenting with different recording techniques, as it improved the week after. 

<div class="img-annotated">
    <img src="/images/comp_investing_efficient_markets.jpg" alt="Definitely coffee in the cup">
    <p>Definitely coffee in the cup</p>
</div>

Lastly, there was a lengthy discussion around portfolio optimisation and the second homework assigment was introduced: performing an Event Study. Most of the code was already provided and only minor tweaks were required to pass the exam.

Week 5 was another extremely useful week for me. Bolliger Bands were introduced, followed by a discussion around the difference between *closing* price and *actual closing* price (actual close factors in things like stock splits and dividends). Additionally, there were modules covered practical techniques for dealing with bad market data from providers. 

The homework here was harder than last week's; creating a straight forward Market Simulator. But, with a bit of effort, I was able to piece something together quickly that passed the quiz. I do question how well I would have done had a human graded my code - it was often not pretty.

Week 6 furthered the discussion around assessing event studies. Then, different investing strategies were compared using [Warren Buffett](http://en.wikipedia.org/wiki/Warren_Buffett) and [Jim Simons](http://en.wikipedia.org/wiki/James_Harris_Simons) as opposing case studies. An in-depth discussion about CAPM followed. As before, I had trouble getting this. Partly, I'm sure, due to a lack of interest.

The week's homework pieced together two of the earlier assigments, creating an event study and then running it through a simulator. Since the hard work was already done, the task was trivial.

Week 7 was, from a financial perspective, probably the most practical week. Videos covered the different information feeds available for hedge fund managers.

The majority of the latter half of the week was set aside to talk about the homework assigment, an implementation of Bollinger Bands.

<div class="img-annotated">
    <img src="/images/comp_investing_question.jpg" alt="An example of the quiz from Week 7">
    <p>An example of the quiz from Week 7</p>
</div>

Week 8, the final module, was centered around two homework assignments building from previous week's work. Where, in the first, we were to perform an event study based on Bollinger Bands and, in the second, feed that data into a market simulator.

I wish I had had more time to spend on them but ended up knocking together a half baked solution in order to pass the exam the moment before I left for a week long holiday. I also needed help from the forum here. I'm not proud of the code, and I doubt I would have done very well if an instructor graded it, but I got it done and passed.

***

<a name="what-i-loved"></a>

## [What I Loved](#what-i-loved)

Firstly, Tucker and the team put a lot of work into the course and it shows. The course was filled with supplimentary material like professional interviews, book recommendations, the QSTK libary, programming tutorials and a wiki rich with content.

I really got a kick out of the programming side too. Just being introduced to NumPy and Pandas made the course worth my time. With the little bit of knowledge I got from this course and [this book](http://shop.oreilly.com/product/0636920023784.do), I was able to rewrite the internals of MagicRanker and make it a shit load faster and more extensibile. For that reason alone, the course was worth my time. 

I was also quite thankful that the homework was relatively easy. Hard enough to keep me thinking but, for a person with a decent amount of programming experience, definitely passable. Perhaps a little too easy at times.

<a name="what-i-would-improve"></a>

## [What I Would Improve](#what-i-would-improve)

The course lectures were sometimes a little dry and, perhaps due to my Gen Y attention span, a bit long. Udacity's model of providing question-based "checkpoints" along the way, really helped to keep me engaged and break up the videos. This course could consider something similar. Come to think of it, there were actually a couple of them in earlier videos but they seemed to disappear by Week 2.

There was also a number of problems with the video quality, including low audio, distorted audio, bloopers and so forth. Sometimes I wondered why they hadn't rerecorded the videos where there were clearly major defects.

I think Coursera has some work to do with the interface too. I found it slightly annoying that the video couldn't be made full screen via the UI. Not sure why they would prevent that, since they have the videos available for download. Lastly, I found the forums a little disorientating. Especially coming from Udacity's forum system where posts can be linked to lessons making it clear what they're about. Though, in fairness, I didn't spend enough time on it.

Completely unrelated to the course quality, but I also had a bit of trouble with a lot of the "maths" that goes into finance. Often there's *theoretical world* values plugged into formulas that involve *real world* implications. The idea of the "risk premium" in CAPM is a good example of this - can the standard deviation of an investment's historic performance really be enough to say if it is safe or not?

<a name="summary"></a>

## [Summary](#summary)

Despite it's minor flaws, this course was really quite phenomenal. If you've got an interest in the quant world and a reasonable programming background, you will almost certainly find this course worthwhile. If you haven't programmed before, then I would consider taking an introductory course first.

Despite the persistant criticisms, the MOOC phenomena over the last couple of years has improved my life considerably and I'm in debt to people, like Tucker and his team, who are willing to upload high-quality courses like this for *free*. We are living in a magical time.
