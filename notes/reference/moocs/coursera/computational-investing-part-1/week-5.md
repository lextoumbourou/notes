---
title: Computational Investing Part I (Coursera) - Week 5
date: 2013-11-27 00:00
status: draft
category: reference/moocs
parent: computational-investing-part-1
tags:
  - MOOC
  - Investing
---

## Week 5

##### 1.1

* Module about data
* Example event
	* Based on Bollinger Bands
	* Price drops below -1.5 standard deviations of recent daily values
	* SPY (S & P 500) is above 0.25
	* Look for examples where individual stock is doing 1 thing, but the market is doing something else
	* Stocks goes from above $5 to below $5
* Survivor bias -- if you only include companies that are alive, you get a better answer
	* Use data that includes "dead" equities

##### 1.2

* **Actual vs Adjusted**
	* Actual: actual closing price recorded by the exchnge on the specific date in history
	* Adjusted: revised price that automatically accounts for "how much you would have if you held the stock"
		* Includes dividends & splits
	* Companies split stocks to gain liquity sometimes
	* Labouring over the point a little much.
	* Didn't need to be 18 mins, imo.
	* I missed it the first time round.
* **Gaps in Data (NaN)
	* Breaks in trading
	* Fill Back
		* Go backwards. When data is missing, add it backwards
		* Fill forward. Go forwards. When data is missing, add it forwards.
		* Always fill forward first. Then, fill backwards to add in data before stock existed.

##### 1.3

* Data sanity and data scrubbing
* Examples of Bad Data
	* Failure to adjust for splits
	* Orders of magnitude drops, followed by offsetting orders of magnitude climbs
	* Database updates missing significant chunks of data/symbols
* Why it's bad
	* You may exploit bad data with automated strategies then fail with real data
	* You think you've "discovered" something but haven't.
* Sanity checks
	* Scan new data for ~50% drops or 200% gains (rare for real data)
	* NaNs in DOW stocks (bad feed data)
	* Recent adjusted prices less than 0.01 -- factor of 10 error
	* NaNs > 20 trading days? -- Something wrong with stock.
* Remove or repair?
	* Easier, more reliable to remove
* Can repair if you have multiple sources

##### 2.1

* Market simulator and event studies

##### 2.2

* Create Python program that accepts orders
