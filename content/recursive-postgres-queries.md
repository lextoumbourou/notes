Title: Understanding recursive PostGres queries
Tags: PostGres, SQL
Date: 2016-03-28
Status: Draft

# Understanding recursive PostGres queries

For some data relationship problems, there's no escaping recursive PostGres queries. For example, let's say you have some categories, and each category can have a sub category, which can have a subcategory and so on. Now, let's say you need to walk from a subcategory deep in the tree to find its parent. As usual, the [PostGres documentation](http://www.postgresql.org/docs/current/static/queries-with.html) is very comprehensive, however, it can be a little tricky to wrap your head around at first. This blog will serve as an attempt to help with the intuition. Hope it helps.

## Step 1. 99 Bottles of Beer in SQL

Firstly the ``VALUES`` query let's you return a single row with a constant value. For example, ``VALUES (99)`` SQL query returns a table with one row: ``99``. Like this:

```
# VALUES (99);
 column1
---------
       99 
(1 row)
```

The ``WITH`` statement lets you build a temporary table which can be used in a subsequent query. So I can rewrite the above query like this:

```
WITH 99_table AS (
   VALUES (99)
)
SELECT * FROM 99_table;
 column1
---------
       99     
(1 row)
```

Obviously not useful so far, but bear with me. Now, what if I wanted to perform an operation on the value in the temporary tables a bunch of times. I don't know, say taking away one from the bottles of beer until there was no more beer? That's where ``WITH RECURSIVE`` comes in. It's called recursive, but it's really not. It's more analgous to a loop. It's broken down into 3 parts:

1. The base query.
2. The union operation, which defines whether to keep or chuck out duplicate rows (more on this).
3. The recursive query, which will keep operating on the base query until no rows are returned.

Let's see if we can write a simple query that starts with our 99 value and counts down to 1:

```
WITH RECURSIVE bottles_of_beer(n) AS (
    VALUES (99)
  UNION ALL
    SELECT n - 1 FROM bottles_of_beer WHERE n > 1
)
SELECT * FROM bottles_of_beer;
```

Let's break it down:

1. ``VALUES (99)``: Start with the base query: ``VALUES (99)``.
2. ``UNION ALL``: Keep "all"; eg don't discard duplicate rows.
3. ``SELECT n - 1 FROM 99_table WHERE n > 1``: Replace ``n`` with ``n - 1`` while ``n`` is greater than 1.

Now finally, using the [``||`` string concatention operator](http://www.postgresql.org/docs/9.1/static/functions-string.html), we can do the bottles of beer song:

```
WITH RECURSIVE bottles_of_beer(n) AS (
    VALUES (99)
  UNION ALL
    SELECT n - 1 FROM bottles_of_beer WHERE n > 1
)
SELECT n::text || ' bottles of beer on the wall' from bottles_of_beer;
 column1
--------------------------------
 99 bottles of beer on the wall
 98 bottles of beer on the wall
 97 bottles of beer on the wall
 96 bottles of beer on the wall
 95 bottles of beer on the wall
# and so on...
```

## Step 2. Walking a tree

Let's try a real world example. Let's say you have some Categories. Then each category can have sub categories and each sub category can have parent categories. That relationship might look like this:

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="299" version="1.1" content="%3CmxGraphModel%20dx%3D%22596%22%20dy%3D%22486%22%20grid%3D%221%22%20gridSize%3D%2210%22%20guides%3D%221%22%20tooltips%3D%221%22%20connect%3D%221%22%20arrows%3D%221%22%20fold%3D%221%22%20page%3D%221%22%20pageScale%3D%221%22%20pageWidth%3D%22826%22%20pageHeight%3D%221169%22%20background%3D%22%23ffffff%22%20math%3D%220%22%20shadow%3D%220%22%3E%3Croot%3E%3CmxCell%20id%3D%220%22%2F%3E%3CmxCell%20id%3D%221%22%20parent%3D%220%22%2F%3E%3CmxCell%20id%3D%222%22%20value%3D%22Shoes%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3BfillColor%3D%23ffffff%3BstrokeColor%3D%23000000%3BlabelBackgroundColor%3D%23ffffff%3BfontStyle%3D1%3BfontSize%3D18%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22250%22%20y%3D%2249%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%223%22%20value%3D%22%26lt%3Bb%26gt%3B%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%2015px%26quot%3B%26gt%3BSports%26lt%3B%2Ffont%26gt%3B%26lt%3B%2Fb%26gt%3B%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22250%22%20y%3D%22140%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%224%22%20value%3D%22Nike%20Air%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3BfontStyle%3D1%3BfontSize%3D13%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22160%22%20y%3D%22230%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%225%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3BexitX%3D0.5%3BexitY%3D0%3BentryX%3D0.5%3BentryY%3D1%3B%22%20edge%3D%221%22%20parent%3D%221%22%20source%3D%224%22%20target%3D%223%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%2210%22%20y%3D%2260%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%2260%22%20y%3D%2210%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%226%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3BexitX%3D0.5%3BexitY%3D0%3B%22%20edge%3D%221%22%20parent%3D%221%22%20source%3D%223%22%20target%3D%222%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%2210%22%20y%3D%2260%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%2260%22%20y%3D%2210%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%227%22%20value%3D%22Rebook%20Pump%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3BfontStyle%3D1%3BfontSize%3D13%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22330%22%20y%3D%22230%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%228%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3BexitX%3D0.5%3BexitY%3D0%3BentryX%3D0.558%3BentryY%3D1%3BentryPerimeter%3D0%3B%22%20edge%3D%221%22%20parent%3D%221%22%20source%3D%227%22%20target%3D%223%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%2210%22%20y%3D%2260%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%2260%22%20y%3D%2210%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2210%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%201%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%22136%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2212%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%202%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22180%22%20y%3D%22230%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2213%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%202%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22350%22%20y%3D%22230%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2214%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%202%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%22180%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2215%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%203%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22180%22%20y%3D%22270%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2216%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%204%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22350%22%20y%3D%22270%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2217%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%201%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%2290%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2218%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%20NULL%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%2245%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3C%2Froot%3E%3C%2FmxGraphModel%3E" ondblclick="(function(svg){if(svg.wnd!=null&amp;&amp;!svg.wnd.closed){svg.wnd.focus();}else{var r=function(evt){if(evt.data=='ready'&amp;&amp;evt.source==svg.wnd){svg.wnd.postMessage(decodeURIComponent(svg.getAttribute('content')),'*');window.removeEventListener('message',r);}};window.addEventListener('message',r);svg.wnd=window.open('https://www.draw.io/?client=1&amp;chrome=0&amp;edit=_blank');}})(this);" viewBox="0 0 299 254" style="max-width:100%;max-height:254px;"><defs><filter id="dropShadow"><feGaussianBlur in="SourceAlpha" stdDeviation="1.7" result="blur"/><feOffset in="blur" dx="3" dy="3" result="offsetBlur"/><feFlood flood-color="#3D4574" flood-opacity="0.4" result="offsetColor"/><feComposite in="offsetColor" in2="offsetBlur" operator="in" result="offsetBlur"/><feBlend in="SourceGraphic" in2="offsetBlur"/></filter></defs><g transform="translate(0.5,0.5)" filter="url(#dropShadow)"><rect x="91" y="5" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(123.5,25.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="54" height="19" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 18px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 56px; white-space: nowrap; font-weight: bold; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;background-color:#ffffff;">Shoes</div></div></foreignObject><text x="27" y="19" fill="#000000" text-anchor="middle" font-size="18px" font-family="Helvetica" font-weight="bold">Shoes</text></switch></g><rect x="91" y="96" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(126.5,117.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="48" height="16" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 48px; white-space: nowrap; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><b><font style="font-size: 15px">Sports</font></b></div></div></foreignObject><text x="24" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><rect x="1" y="186" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(35.5,209.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="50" height="13" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 13px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 50px; white-space: nowrap; font-weight: bold; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;">Nike Air</div></div></foreignObject><text x="25" y="13" fill="#000000" text-anchor="middle" font-size="13px" font-family="Helvetica" font-weight="bold">Nike Air</text></switch></g><path d="M 61 186 L 144.96 158.01" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 149.94 156.35 L 144.41 161.89 L 144.96 158.01 L 142.19 155.25 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 151 96 L 151 71.37" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 151 66.12 L 154.5 73.12 L 151 71.37 L 147.5 73.12 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><rect x="171" y="186" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(186.5,209.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="88" height="13" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 13px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 88px; white-space: nowrap; font-weight: bold; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;">Rebook Pump</div></div></foreignObject><text x="44" y="13" fill="#000000" text-anchor="middle" font-size="13px" font-family="Helvetica" font-weight="bold">Rebook Pump</text></switch></g><path d="M 231 186 L 163.89 158.42" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 159.03 156.42 L 166.84 155.85 L 163.89 158.42 L 164.18 162.32 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><g transform="translate(128.5,95.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="45" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 45px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = 1</font></div></div></foreignObject><text x="23" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(38.5,189.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="45" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 45px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = 2</font></div></div></foreignObject><text x="23" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(208.5,189.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="45" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 45px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = 2</font></div></div></foreignObject><text x="23" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(141.5,139.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 2</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(51.5,229.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 3</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(221.5,229.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 4</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(141.5,49.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 1</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(120.5,4.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="61" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 61px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = NULL</font></div></div></foreignObject><text x="31" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g></g></svg>

Let's start by creating that table.

```
CREATE TABLE category (
    id integer,
    name varchar(256),
    parent_id integer,
    PRIMARY KEY(id)
);
```

Next, we'll add the categories in the diagram:

```
INSERT INTO category (id, name, parent_id) VALUES (1, 'Shoes', NULL);
INSERT INTO category (id, name, parent_id) VALUES (2, 'Sports', 1);
INSERT INTO category (id, name, parent_id) VALUES (3, 'Nike Air', 2);
INSERT INTO category (id, name, parent_id) VALUES (4, 'Reebok Pumps', 2);
```

Okay, first challenge: starting with Reebok Pumps, can we find all parent categories?

Following on from the earlier breakdown, first define the non-recursive part:

```
SELECT id, name, parent_id FROM categories WHERE id = 3;
```

Now we decide on the union operator, in this example I'll use ``UNION`` to remove duplicate rows. Lastly, we define the recursive bit, we'll continue to select from ``start_category`` until no rows are returned:

```
WITH RECURSIVE start_category AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 3
    UNION
    SELECT category.id, category.name, category.parent_id
        FROM category
        JOIN start_category ON (category.id = start_category.parent_id)
)
SELECT * FROM start_category;
 id |     name     | parent_id
----+--------------+-----------
  3 | Rebook Pumps |         2
  2 | Sports       |         1
  1 | Shoes        |
(3 rows)
```

Cool. That seems to work. Okay, challenge number 2: let's go the other way, start at the top-level and return all the children.

```
WITH RECURSIVE top_level AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 1
    UNION ALL
    SELECT category.id, category.name, category.parent_id
         FROM category
         JOIN top_level ON (category.parent_id = top_level.id)
)
SELECT * FROM top_level;
```

Okay, so going back to the first example what would happen if someone goofed and our top-level category pointed to a low-level category. Effectively creating a circulare relationship. Like this:

Let's try it:

```
UPDATE category SET parent_id = 3 WHERE id = 1;
```

Then run the same query:

```
WITH RECURSIVE start_category AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 3
    UNION ALL
    SELECT category.id, category.name, category.parent_id
        FROM category
        JOIN start_category ON (category.id = start_category.parent_id)
)
SELECT * FROM start_category;
```

Notice how the query never stops? That's because each time the recursive part of the query is run, rows are returned. So it keeps going.

We can combat this problem by using ``UNION`` instead of ``UNION ALL`` which discards duplicate rows:

```
WITH RECURSIVE start_category AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 3
    UNION
    SELECT category.id, category.name, category.parent_id
        FROM category
        JOIN start_category ON (category.id = start_category.parent_id)
)
SELECT * FROM start_category;
 id |     name     | parent_id
----+--------------+-----------
  3 | Rebook Pumps |         2
  2 | Sports       |         1
  1 | Shoes        |         3
(3 rows)
```

Cool, huh?

### How it works.

Okay, so let's try to understand what's going on here, according to the docs.

The first part is to evaluate the non-recursive term. It's then put into a "working" table and what's returned is appended to our results. So we have effectively 2 data structres:

```
SELECT id, name, parent_id FROM category WHERE id = 3;
```

```
Results                               Working table

3   rebook pumps   2                  3  rebook pumps 2
```

Then we evaluate the recursive term against the working table, which in this case is ``SELECT category.id, category.name, category.parent_id FROM category JOIN start_category ON (category.id = start_category.parent_id)``. So basically, get something from ``category`` which can be joined against our working table. So, we fetch the sports category and this time add it to a "Intermediate table".

Intermediate table:

```
id  |   name           | parent_id
----+--------------+-----------
 2  |   Sports         | 1
```

Then this table replaces and working table and the result is appended to results. So we look like this:

```
Results                                Working table

id  |   name           | parent_id       id  |   name           | parent_id
----+--------------------------        -------------------------------------
 3  |   Rebook Pumps   | 2                2  |   Sports         | 1
 2  |   Sports         | 1             
```

Then we go again until nothing is in the working table.
