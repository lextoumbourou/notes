# Lesson 2: Optimizing the CRP

* PageSpead is fucking awesome: http://developers.google.com/speed/pagespeed/insights/

* Optimizing the DOM
	* Minify assets
		* Remove comments, shortern variables etc
	* Compress
		* GZip
			* All modern browsers support it and will auto negotiate it
			* Ensure your server has it configured
	* Cache
		* Validation token
			* In request Etag HTTP header 
			* If data hasn't changed, then no data transfer occurs
		* Use Cache-Control header
* Unblocking CSS
	* Browser renders the page only once it has the CSS and builds the CSSOM
	* Consider using media queries in HTML markup to stop render blocking on certain media queries
		```
		<link rel="stylesheet" href="style-print.css" media="print">
		```
		* Note: stylesheet is still download, just doesn't block rendering
* External Javascript dependancies
	* JavaScript is parser-blocking.
	* If you are requesting external JavaScript files before your HTML has finished rendering, you can slow DOM rendering.
		* Best practise: Push JavaScript loads to the bottom of the ```<body></body>``` block.
		* For JavaScript that doesn't modify the DOM / need access to CSSOM, use the ```async``` attribute on the script tag.
			* For example: this won't block the parser:
			```<script src="analytics.js" async></script>```
* Blocking / inline / async recap
	* Blocking
		* Has to wait for CSS to download completely before running JavaScript
	* Inline
		* Doesn't have to request JS resource remotely: can be faster
	* async
		* Requests the JS resource without block parser. Cannot modify the DOM.
* General Strategies recap
	* Minimize / compress / cache
	* Minimize use of render blocking resources (CSS)
		* Use media queries on ```<link>`` to unblock resources
		* Inline CSS (though probably a bad idea for tidiness purposes)
	* Minimize use of parser blocking resources (JS)
		* Defer JavaScript execution - move to end of body
		* Use async attribute on ```<script>``` if script doesn't modify the DOM
* Minimize Critical Rendering Path resources
	* Minimize size and amount of HTML/CSS/JavaScript files
	<img src="images/CRP-metrics.png"></img>
* [Preload Scanner](http://andydavies.me/blog/2013/10/22/how-the-browser-pre-loader-makes-pages-load-faster/)
	* Browser process that "peaks ahead" in the document to discover blocking JS and CSS, then downloads them while parser is blocked
* CRP Diagram notes:

```
[Example 2](https://github.com/igrigorik/udacity-webperf/blob/master/assets/ex2-diagram.html) diagram

1. Page requests HTML, browser is idle until we get response.
2. Server responds with content. HTML parsing begins, as does DOM construction.
3. Parser gets to ```css``` elements. Downloads them simultaneously while DOM constructions continues - rendering is blocked until the ```style.css``` resource is downloaded. ```print.css``` does not block building Render Tree because the media query is for print only..
4. DOM parsing continues until it reaches ```app.js``` script. Parser is blocked until ```app.js``` is downloaded.
5. Response for ```app.js``` occurs, parser continues until it hits ```analytics.js```. Parser and rendering are not blocked as request occurs asynchrously.
6. Response for css elements occurs and CSSOM is constructed.
7. Lastly, Render Tree is constructed and we move to Render, Layout then Paint step.
```
