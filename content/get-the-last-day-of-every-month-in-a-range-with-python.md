Title: Get the last day of every month in a range with Python
Tagline: A problem I solved by stealing from Stack Overflow
Slug: get-the-last-day-of-every-month-in-a-range-with-python
Date: 2012-02-23

</p>

<div class="intro">
I'm building an application that determines what your stock portfolio
was worth every month, factoring in brokerage costs and interest rates,
and compares it to the amount you paid for it (your cost basis). To do
this, I needed to write a method that iterates over a date range and
spits back the last day of each month, then builds a query based on
that.
</div>

I had a few rough ideas on how to go about it but most of them ending up
looking ugly enough to make Baby Jesus weep. People on Stack Overflow
had some ideas, most, of course, better than mine. However, I couldn't
find any that were easy to read and understand.

So, the method I went with was an amalgamation of Vin-G's [idea][] - who
I'm pretty sure makes his mother proud - and some stuff that came out of
reading up on Generators.

It looks like this:

    :::python
    def months_in_range(start_date, end_date):
        """Get the last day of every month in a range between two datetime values.
        Return a generator
        """
        start_month = start_date.month
        end_months = (end_date.year-start_date.year)*12 + end_date.month

        for month in range(start_month, end_months + 1):
            # Get years in the date range, add it to the start date's year
            year = (month-1)/12 + start_date.year
            month = (month-1) % 12 + 1

            yield datetime(year, month, 1)-timedelta(days=1))

The method is called with two date time instances: a start and end date
and is then iterated when called. Like so:

    :::python
    for date in months_in_range(datetime(2006, 05, 01), datetime.now()):
        get_cost_basis(date)

I also used this code to do the same thing but a return a generator of
years:

    :::python
    def year_iterator(start_date, end_date):
        output = []
        for year in range(start_date.year, end_date.year+1):
            yield datetime(year, 12, 31)


[Hooray!][]

  [idea]: http://stackoverflow.com/questions/4039879/best-way-to-find-the-months-between-two-dates-in-python
  [Hooray!]: http://www.twitter.com/lexandstuff
  [comments powered by Disqus.]: http://disqus.com/?ref_noscript
  [comments powered by <span class="logo-disqus">Disqus</span>]: http://disqus.com
