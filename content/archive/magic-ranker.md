Title: Magic Ranker
Tagline: An ASX implementation of Joel Greenblatt's Magic Formula
Slug: magic-ranker
Date: 2011-07-31
Tags: Python, Django

### Magic Formula

I finished reading Joel Greenblatt's [The Little Book That Beats The
Market](http://www.amazon.com/Little-Book-That-Beats-Market/dp/0471733067) a while back and thought it was excellent. Basically, it's
value investing à la Warren Buffett, Benjamin Graham explained through
easy analogy. Joel does an amazing job of explaining a complex idea such
that a 12 year old could understand (or even me!). The premise, of
course, is to *buy great companies at bargain rates*. Joel describes a
formula toward the end of the book that provides a strategy for listing
these companies and buying them annually. Finally, if that wasn't simple
enough for you (how simple do you need it exactly?), Joel runs a site
[MagicFormulaInvesting.com](MagicFormulaInvesting.com) that lists these companies and even offers
a [fund](http://http//www.formulainvesting.com/) that manages it all for you.

This is awesome if you're in North America and investing in NASDAQ, but
not so if your house is upside down in Australia. Hence, [MagicRanker][].

### Magic Ranker

[![image][]]  

MagicRanker, in brief, is what's in the book but for the ASX plus some
additives and artificial colours. It'd probably be easy to extend to
other markets around the world, if anyone is interested...we'll see. The
heart of it is a simple Python/Django app that scrapes Yahoo Finance and
E-trade for data, then ranks it nightly. Like so:

1.  Ranks the companies by ROE (Return On Equity) from highest to lowest
    (I eliminated those over 70% as I assumed some accounting trickery
    was going on there).
2.  Ranks the companies by Price to Earnings (P/E) lowest to highest.
3.  Adds those ranks together to get the total rank.

I've added a couple of features like having the average ROE over a 5 /
10 year period, to get the most consistenly excellent companies (but of
course they're not necessarily the cheapist) and a debt percentage
filter. Feel free to ignore those options.

Lastly, I automated the whole thing with a couple of cronjobs:

-   **Daily cron** scrapes the web for ROE, P/E and ranks the stocks.  
-   **Weekly cron** scraps the ASX for new companies.
-   **Monthly cron** get historical balance sheet data (since it's only
    updated twice a year).

The front end code uses [YUI][] for the nifty buttons and the
[Blueprint][] CSS framework for the grid.

### Disclaimer (If you lose all your money, it's not my fault)

Keep in mind, I did this for fun and experimentation, not as a serious
financial tool. I'm not sure if I'll invest real money yet, might just
setup an E-trade watchlist and see how it goes. You should consider
doing that too.

So, here it is: [MagicRanker.com](http://MagicRanker.com).

  [MagicRanker]: http://www.magicranker.com
  [image]: /images/magic_ranker_home.png
  [![image][]]: http://magicranker.com
  [1]: /images/magic_ranker_example.png
  [![image][1]]: http://magicranker.com
  [YUI]: http://developer.yahoo.com/yui/
  [Blueprint]: http://blueprintcss.org/
