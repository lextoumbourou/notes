---
title: Computational Investing Part I (Coursera) - Week 1
date: 2013-11-27 00:00
status: draft
category: reference/moocs
parent: computational-investing-part-1
tags:
  - MOOC
  - Investing
---

## Week 1

### Module 1

* In the overview it mentions students will grade each other - though that was incorrect (unless I missed something!)
* A few books are recommended though I'm not sure I'm interested enough to pick them up.

### Module 2

* Basics for hedge fund managers -- computation is barely touch on

#### 2.1

* Incentives for portfolio managers discussed
* Attracting investors
  * general info about the types of investors and how to attract them
* Types of fund goals

#### 2.2

* Common metrics for funds
  * Annual return
  * Risk (calculated as standard deviation of return)
  * Risk (draw down?)
  * Reward/Risk: Sharpe ratio
  * Sortino Ratio -- only counts risk when it goes down
* Calculating Annual return
  * `metric = (value[end] / value[start]) - 1 = (100 / 50) = 1 = 100%`
* Standard deviation of daily return
  * `daily_rets[i] = (value[i]/value[i-1]) - 1 # where i is the day of the year`
  * `std_metric = stdev(daily_rets)`
* Max draw down -- not really described in detail

#### 2.3

* Sharpe ratio
  * Used to compare "risk" of similar portfolios
  * higher sharpe ratio == more return for the same risk
  * Reward/Risk
  * `metric = k * mean(daily_rets)/stdev(daily_rets))`
  * `k = sqrt(250)` for daily returns

#### 2.4

* Learn how to calculate metrics from last two modules
* Adjusted close -- includes dividends -- covered in depth later

### Module 3

#### 3.1

* Types of orders
  * Buy, Sell
  * Market, Limit
  * Shares
  * Price (if limit)
* Understanding order book at exchanges
  * Crossing the spread == when somebody with a bid pays the asking price

#### 3.3

* Exploiting the market as a hedge fund manager
  * Order book observation (needs low latency)
  * Arbitrage (when markets go out of sync -- very rarely works)

#### 3.4

* Mechanics of the Market
* Overview of how computing works inside a hedge fund
* News packets

### Module 4

#### 4.1

* Interview with Paul Jiganti -- trader
* A market maker?
* Fairly difficult to follow
* Dark pools?

#### 4.2

* How orders flow through the system?
