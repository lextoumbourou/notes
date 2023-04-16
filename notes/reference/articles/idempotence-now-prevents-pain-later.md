---
title: Idempotence Now Prevents Pain Later
date: 2021-04-06 00:00
category: reference/articles
status: draft
---

Notes from article [Idempotence Now Prevents Pain Later](https://ericlathrop.com/2021/04/idempotence-now-prevents-pain-later) by Eric Lathrop

* Idempotence is the property that when software is run more than once, it only has the effect of running once.
* Simple example:
	* Run a cron job that charges customers every month
* Without idempotence, if that went wrong and failed halfway through, it might be a nightmare to clean up the mess, especially if it was halfway through.
* Improved example:
	* Run cron job every hour that checks for accounts that haven't been charged this month.
