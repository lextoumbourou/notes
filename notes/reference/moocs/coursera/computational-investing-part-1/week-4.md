---
title: Computational Investing Part I (Coursera) - Week 4
date: 2013-11-27 00:00
status: draft
category: reference/moocs
parent: computational-investing-part-1
tags:
  - MOOC
  - Investing
---

## Week 4

#### 1.1

* Sound was terrible
* Ray Charles got a big annoying
* Quantitive investment is based on 'arbitrage' model
	* Discrepency between price and value
* Calculating value
	* Technical analysis - pricing and volume
	* Fundamental analysis
		* financial statements
		* P/E ratios
* Where information comes from:
	* Price/Volume: the markets
	* Fundamentals: SEC fillings

#### 1.2

* Efficient markets hypothesis
	* Weak version
		* future prices cannot be predicted by analysing prices from the past
	* Semi-strong
		* no room for arbitrage
	* Strong
		* market is efficient.
		* Prices reflect *hidden* infroamtion
* Some evidence that it's not true
* Behaviour economies is an argument against EMH

#### 1.3

* Again, terrible sound
* Study of how positive and negative event affected prices of related stocks
* Drift up and down for bad news before the event
	* Why? Leakage can occur.
	* Bad news can just suggest a bad stock and vice versa
* Upcoming assignment talking about running and event study
* Align stocks so Day 0 is the same for all of them (??)

#### 2.1

* Terrible sound - microphone was way too loud
* Understand "risk", correlation and covariance, mean variance optimization and efficient frontier
* Portfolio optimizer
	* Given: set of equities and target return, what's the optimal portfolio?
	* Find: allocation to each equity that minimizes risk
* Risk often refers to how "volatile" their stock is. Volatility is calculated by standard deviation
* Ideal portfolio is high return and low risk but that's rare. You want a healthy combination - each are a trade off.
* Harry Markowitz developed "mean variance optimisation" theory.

#### 2.2 - Inputs and Outputs of a Portfolio Optimizer

* Portfolio optimizer balances return and risk. Exploits information about it.
* Inputs
	* Expected return for each equity
	* Volatility (risk) for each equity
	* Target return
	* Covariance matrix -- no idea what this is?
* Output -- optimal portfolio

##### Opinion

* Can't help but feel that a portfolio optimizer is a silly idea. You make returns based on future earnings which you can't really "optimise" for.

#### 2.3

* How can we have a portfolio with lower risk than individual equities?
* Higher weight to "lower risk" stocks. Look at covariance and anti correlated stocks.

#### 2.4

* Efficient frontier -- Lower risk by combining anti-correlated equities
* How to combine them

#### 2.5

* How Optimizer Works
	* Define variables
		* Things optimiser can "tweak
		* Vary how much to allocate to equities (weights)
	* Define constraints
		* Sum of all weights must add up to 1
		* No less than 10% in a certain equity
	* Define optimization criteria
* Optimizer algorithm
	1. Tweak weights
	2. Check constraints
	3. OK?
	4. Call function
	5. Repeat
* Could be a giant ```for``` loop that brute forces the thing.
* QSTK uses an optimizer called CVXOPT

##### 3.1

* Overview of what event studies are
* Slow movies. Lot's of static

##### 3.2

* Found it a lot easier to understand most of the lesson after completing the course work. I guess that's standard.
* When a stock drops below $5 it changes a lot of things
	* Not listed on major indicies

## Assignment

* Fairly easy
* Most of the data was already provided
