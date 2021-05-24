## Week 7

##### 1.1

* Information feeds -- for event studies
	* Thomson Reuters Machine Readable News
		* Machine readable news feeds	 
	* StarMine
	* InsiderInsights
* What to look for
	* Historical data you can back test
	* Survivor bias free
	* Ease of integration with your sys
	* Low latency
		* Get to you quickly so you can act on it. 

##### 1.2

* TA uses historical price and volume data *only* to compute "indicators"
* Indicators are calculated from recent price and volume data to predict future price movements
* Indicators are "heuristics"
* Discusses the controversial side of technical analysis
	* An information source? 
	* Depends on info in historical price and volume
* "Market physics"
	* Limit to how quickly market can move
* Tucker's view on Technical Analysis is interesting

##### 1.3

* How to compute Bollinger Bands
	* Read in historical closing prices
	* mid = "rolling" mean over look back period (average over 5 week period, for example)
	* std = "rolling" STDEV over look back
	* upper = mid + std
	* lower = mid - std 
* Current "value" of Bollinger
	* if current price at Upper band = 1.0
	* if current price at Lower band = -1.0
* Val = (price - mid / std)   


