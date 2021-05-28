Link: https://ericlathrop.com/2021/04/idempotence-now-prevents-pain-later/
Author: Eric Lathrop
Published: 2012-4-6
Tags: #Data

---

* Idempotence is the property that when software is run more than once, it only has the effect of running once.
* Simple example:
	* Run a cron job that charges customers every month
* Without idempotence, if that went wrong and failed halfway through, it might be a nightmare to clean up the mess, especially if it was half way through.
Improved example:
	* Run cron job every hour that checks for accounts that haven't been charged this month.