---
title: Recursive queries in PostgreSQL
date: 2016-05-08 00:00
tags:
  - PostgreSQL
  - DataEngineering
---

For some data relationships in Postgres (or any other relational database that speaks SQL), recursive queries are near inevitable. Let's say you have some top-level product categories, like Shoes, Hats, Wigs etc, and each of those has subcategories, like Sports, Casual, Formal etc, which can have their own subcategories and so on. Something like this:

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="299" version="1.1" content="%3CmxGraphModel%20dx%3D%22706%22%20dy%3D%22713%22%20grid%3D%221%22%20gridSize%3D%2210%22%20guides%3D%221%22%20tooltips%3D%221%22%20connect%3D%221%22%20arrows%3D%221%22%20fold%3D%221%22%20page%3D%221%22%20pageScale%3D%221%22%20pageWidth%3D%22826%22%20pageHeight%3D%221169%22%20background%3D%22%23ffffff%22%20math%3D%220%22%20shadow%3D%220%22%3E%3Croot%3E%3CmxCell%20id%3D%220%22%2F%3E%3CmxCell%20id%3D%221%22%20parent%3D%220%22%2F%3E%3CmxCell%20id%3D%222%22%20value%3D%22Shoes%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3BfillColor%3D%23ffffff%3BstrokeColor%3D%23000000%3BlabelBackgroundColor%3D%23ffffff%3BfontStyle%3D1%3BfontSize%3D18%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22250%22%20y%3D%2249%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%223%22%20value%3D%22%26lt%3Bb%26gt%3B%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%2015px%26quot%3B%26gt%3BSports%26lt%3B%2Ffont%26gt%3B%26lt%3B%2Fb%26gt%3B%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22250%22%20y%3D%22140%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%224%22%20value%3D%22Nike%20Air%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3BfontStyle%3D1%3BfontSize%3D13%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22160%22%20y%3D%22230%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%225%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3BexitX%3D0.5%3BexitY%3D0%3BentryX%3D0.5%3BentryY%3D1%3B%22%20parent%3D%221%22%20source%3D%224%22%20target%3D%223%22%20edge%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%2210%22%20y%3D%2260%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%2260%22%20y%3D%2210%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%226%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3BexitX%3D0.5%3BexitY%3D0%3B%22%20parent%3D%221%22%20source%3D%223%22%20target%3D%222%22%20edge%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%2210%22%20y%3D%2260%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%2260%22%20y%3D%2210%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%227%22%20value%3D%22Rebook%20Pump%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3BfontStyle%3D1%3BfontSize%3D13%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22330%22%20y%3D%22230%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%228%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3BexitX%3D0.5%3BexitY%3D0%3BentryX%3D0.558%3BentryY%3D1%3BentryPerimeter%3D0%3B%22%20parent%3D%221%22%20source%3D%227%22%20target%3D%223%22%20edge%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%2210%22%20y%3D%2260%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%2260%22%20y%3D%2210%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2210%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%201%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%22136%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2212%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%202%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22180%22%20y%3D%22230%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2213%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%202%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22350%22%20y%3D%22230%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2214%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%202%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%22180%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2215%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%203%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22180%22%20y%3D%22270%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2216%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%204%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22350%22%20y%3D%22270%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2217%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%201%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%2290%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2218%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%20NULL%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%2245%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3C%2Froot%3E%3C%2FmxGraphModel%3E" ondblclick="(function(svg){if(svg.wnd!=null&amp;&amp;!svg.wnd.closed){svg.wnd.focus();}else{var r=function(evt){if(evt.data=='ready'&amp;&amp;evt.source==svg.wnd){svg.wnd.postMessage(decodeURIComponent(svg.getAttribute('content')),'*');window.removeEventListener('message',r);}};window.addEventListener('message',r);svg.wnd=window.open('https://www.draw.io/?client=1&amp;chrome=0&amp;edit=_blank');}})(this);" viewBox="0 0 299 254" style="max-width:100%;max-height:254px;"><defs><filter id="dropShadow"><feGaussianBlur in="SourceAlpha" stdDeviation="1.7" result="blur"/><feOffset in="blur" dx="3" dy="3" result="offsetBlur"/><feFlood flood-color="#3D4574" flood-opacity="0.4" result="offsetColor"/><feComposite in="offsetColor" in2="offsetBlur" operator="in" result="offsetBlur"/><feBlend in="SourceGraphic" in2="offsetBlur"/></filter></defs><g transform="translate(0.5,0.5)" filter="url(#dropShadow)"><rect x="91" y="5" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(123.5,25.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="54" height="19" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 18px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 56px; white-space: nowrap; font-weight: bold; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;background-color:#ffffff;">Shoes</div></div></foreignObject><text x="27" y="19" fill="#000000" text-anchor="middle" font-size="18px" font-family="Helvetica" font-weight="bold">Shoes</text></switch></g><rect x="91" y="96" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(126.5,117.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="48" height="16" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 48px; white-space: nowrap; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><b><font style="font-size: 15px">Sports</font></b></div></div></foreignObject><text x="24" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><rect x="1" y="186" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(35.5,209.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="50" height="13" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 13px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 50px; white-space: nowrap; font-weight: bold; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;">Nike Air</div></div></foreignObject><text x="25" y="13" fill="#000000" text-anchor="middle" font-size="13px" font-family="Helvetica" font-weight="bold">Nike Air</text></switch></g><path d="M 61 186 L 144.96 158.01" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 149.94 156.35 L 144.41 161.89 L 144.96 158.01 L 142.19 155.25 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 151 96 L 151 71.37" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 151 66.12 L 154.5 73.12 L 151 71.37 L 147.5 73.12 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><rect x="171" y="186" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(186.5,209.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="88" height="13" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 13px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 88px; white-space: nowrap; font-weight: bold; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;">Rebook Pump</div></div></foreignObject><text x="44" y="13" fill="#000000" text-anchor="middle" font-size="13px" font-family="Helvetica" font-weight="bold">Rebook Pump</text></switch></g><path d="M 231 186 L 163.89 158.42" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 159.03 156.42 L 166.84 155.85 L 163.89 158.42 L 164.18 162.32 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><g transform="translate(128.5,95.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="45" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 45px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = 1</font></div></div></foreignObject><text x="23" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(38.5,189.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="45" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 45px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = 2</font></div></div></foreignObject><text x="23" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(208.5,189.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="45" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 45px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = 2</font></div></div></foreignObject><text x="23" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(141.5,139.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 2</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(51.5,229.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 3</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(221.5,229.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 4</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(141.5,49.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 1</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(120.5,4.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="61" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 61px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = NULL</font></div></div></foreignObject><text x="31" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g></g></svg>

Then, you wish to construct a query where for some child category, you want to fetch the entire category hierarchy. For example, starting at **Nike Air** walk up the tree to find all parent categories (**Sports** -> **Shoes**). Or maybe you want to go the other way, starting at **Shoes** find all child categories. In either scenario, you're going to want to understand recursive queries.

*Side note: in this contrived example, there are some techniques you could use to skip recursive queries completely, like denormalising your data by adding the parent id into each child category which would also bring significant performance gains, but I still think it's useful to have recursive queries in your toolkit...*

As always, the [Postgres docs](http://www.postgresql.org/docs/current/static/queries-with.html) are very comprehensive, however, they can be a little daunting when you're getting started - especially if writing SQL isn't something you're doing every day. Since I was recently tasked with solving a similar problem, the difficulties of understanding it are fresh on my mind. So I figured I'm in a good position to help others do the same.

***

First things first, you'll need to...

## Understand ``WITH`` queries

``WITH`` queries lets you build a temporary table for use in a subsequent query, also known as "Common Table Expressions" or CTEs. Here's an example where I build a table which uses the [VALUES](http://www.postgresql.org/docs/9.1/static/sql-values.html) command to return a single row with a number:

```
WITH bottles_of_beer AS (
   VALUES (99)
)
SELECT * FROM bottles_of_beer;
 column1
---------
       99     
(1 row)
```

*Side note: This example doesn't do justice to the usefulness of the ``WITH`` statement check out blogs posts like [this](http://www.craigkerstiens.com/2013/11/18/best-postgres-feature-youre-not-using/) for cool stuff you can do with it.*

***

Now what if I wanted to perform an operation on the value returned in the temporary table, like taking away 1 from our bottles of beer? And, what if I then wanted to perform an operation on the output of *that* operation and so on? Think you can see where I'm going with this.

For that, we'll need to...

## Understand ``WITH RECURSIVE`` queries

A ``WITH RECURSIVE`` query let's your perform an operation on the output of a ``WITH`` statement until there are no more rows to return. The query is broken down into 3 parts:

1. The base query: the query that runs first.
2. The ``UNION`` operator: which says whether we'll keep duplicate rows or not.
3. Then, the recursive part: which will keep operating on the last query's output until no rows are returned or we hit our end condition.

Let's see if we can use that information to rewrite a query that counts down from 99 to 1:

```
WITH RECURSIVE bottles_of_beer(n) AS (              -- See #1
    VALUES (99)                                     -- See #2
  UNION ALL                                         -- See #3
    SELECT n - 1 FROM bottles_of_beer WHERE n > 1   -- See #4
)
SELECT * FROM bottles_of_beer;
```

Let's break it down:

1. Create a temporary table, aka CTE, called ``bottles_of_beer`` with a single column referred to as ``n``.
2. Build the base query: ``VALUES (99)`` which returns the number ``99``.
3. Tell Postgres what we want to do with duplicate rows. In this query we're keeping "all", ie not discarding duplicate rows (more on that later).
4. Then we define the recursive query, which will subtract 1 from ``n`` while ``n`` is greater than 1.

Finally, we can use the [``||`` string concatention operator](http://www.postgresql.org/docs/9.1/static/functions-string.html), to do the whole bottles of beer song:

```
WITH RECURSIVE bottles_of_beer(n) AS (
    VALUES (99)
  UNION ALL
    SELECT n - 1 FROM bottles_of_beer WHERE n > 1
)
SELECT n::text || ' bottles of beer on the wall' from bottles_of_beer;
```

```
 column1
--------------------------------
 99 bottles of beer on the wall
 98 bottles of beer on the wall
 97 bottles of beer on the wall
 96 bottles of beer on the wall
 95 bottles of beer on the wall
# and so on...
```

Cool!

***

Armed with that knowledge, we can head back to the original problem. But first, it'll help if we...

## Understand ``JOIN``s in ``WITH RECURSIVE`` queries

Joins in ``WIH RECURSIVE`` queries can be used to join data from a table against the current output of the ``WITH`` query. We can see that in action by solving the problem described in the intro: **starting with Nike Air, can we walk up the tree finding all parent categories?**

Firstly let's create the table to represent our data relationship:

```
CREATE TABLE category (
    id integer,
    name varchar(256),
    parent_id integer,
    PRIMARY KEY(id)
);
```

Next, we'll add the example categories in our diagram:

```
INSERT INTO category (id, name, parent_id) VALUES
    (1, 'Shoes', NULL),
    (2, 'Sports', 1),
    (3, 'Nike Air', 2),
    (4, 'Reebok Pumps', 2);
```

Since we have learned the basics of building recursive queries, we can firstly define the non-recursive part:

```
SELECT id, name, parent_id FROM categories WHERE id = 3;
```

which simply grabs the ``Nike Air`` category by id.

Let's put that in a CTE:

```
WITH RECURSIVE category_tree AS (
   SELECT id, name, parent_id FROM categories WHERE id = 3
)
```

Now we decide on the union operator, in this example I'll use ``UNION`` to remove duplicate rows.

Now here's where it gets interesting: defining the recursive part. Here we'll join the output from the base query to the ``category`` table until there's nothing more to be joined:

```
    SELECT category.id, category.name, category.parent_id
        FROM category
        JOIN category_tree ON (category_tree.parent_id = category.id)
```

So, when the ``Nike Air`` category is first returned, we'll grab a row from the ``category`` table which matches it's parent id. Then when that comes back, we'll grab its parent until we can't find any more parents.

Now, let's put it all together:

```
WITH RECURSIVE category_tree AS (
   SELECT id, name, parent_id FROM category WHERE id = 3
     UNION
   SELECT category.id, category.name, category.parent_id
        FROM category
        JOIN category_tree ON (category.id = category_tree.parent_id)
)
SELECT * FROM category_tree;
```

Go ahead and run that. Does it work? Here's what was returned for me:

```
 id |     name     | parent_id
----+--------------+-----------
  3 | Nike Air     |         2
  2 | Sports       |         1
  1 | Shoes        |
(3 rows)
```

Cool!

***

Okay, challenge number 2, let's go the other way: **start at the top-level return all the children**.

There's nothing new here, we just simply change our base query to start by selecting our top-level parent then join any category with a parent id that matches our current result's id. Like so:

```
WITH RECURSIVE top_level AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 1
      UNION
    SELECT category.id, category.name, category.parent_id
         FROM category
         JOIN top_level ON (category.parent_id = top_level.id)
)
SELECT * FROM top_level;
```

Go ahead and run that. I'll wait...

Here's what was returned for me:

```
SELECT * FROM top_level;
 id |     name     | parent_id
----+--------------+-----------
  1 | Shoes        |
  2 | Sports       |         1
  3 | Nike Air     |         2
  4 | Reebok Pumps |         2
(4 rows)
```

Choice!

***

Now what would happen if someone goofed up and created a top-level category with a parent id of a child category, creating a circular relationship. Like this:

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="336" version="1.1" content="%3CmxGraphModel%20dx%3D%22650%22%20dy%3D%22486%22%20grid%3D%221%22%20gridSize%3D%2210%22%20guides%3D%221%22%20tooltips%3D%221%22%20connect%3D%221%22%20arrows%3D%221%22%20fold%3D%221%22%20page%3D%221%22%20pageScale%3D%221%22%20pageWidth%3D%22826%22%20pageHeight%3D%221169%22%20background%3D%22%23ffffff%22%20math%3D%220%22%20shadow%3D%220%22%3E%3Croot%3E%3CmxCell%20id%3D%220%22%2F%3E%3CmxCell%20id%3D%221%22%20parent%3D%220%22%2F%3E%3CmxCell%20id%3D%222%22%20value%3D%22Shoes%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3BfillColor%3D%23ffffff%3BstrokeColor%3D%23000000%3BlabelBackgroundColor%3D%23ffffff%3BfontStyle%3D1%3BfontSize%3D18%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22250%22%20y%3D%2249%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%223%22%20value%3D%22%26lt%3Bb%26gt%3B%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%2015px%26quot%3B%26gt%3BSports%26lt%3B%2Ffont%26gt%3B%26lt%3B%2Fb%26gt%3B%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22250%22%20y%3D%22140%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%224%22%20value%3D%22Nike%20Air%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3BfontStyle%3D1%3BfontSize%3D13%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22160%22%20y%3D%22230%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%225%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3BexitX%3D0.5%3BexitY%3D0%3BentryX%3D0.5%3BentryY%3D1%3B%22%20parent%3D%221%22%20source%3D%224%22%20target%3D%223%22%20edge%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%2210%22%20y%3D%2260%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%2260%22%20y%3D%2210%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%226%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3BexitX%3D0.5%3BexitY%3D0%3B%22%20parent%3D%221%22%20source%3D%223%22%20target%3D%222%22%20edge%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%2210%22%20y%3D%2260%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%2260%22%20y%3D%2210%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%227%22%20value%3D%22Rebook%20Pump%22%20style%3D%22rounded%3D1%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3BfontStyle%3D1%3BfontSize%3D13%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22330%22%20y%3D%22230%22%20width%3D%22120%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%228%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3BexitX%3D0.5%3BexitY%3D0%3BentryX%3D0.558%3BentryY%3D1%3BentryPerimeter%3D0%3B%22%20parent%3D%221%22%20source%3D%227%22%20target%3D%223%22%20edge%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%2210%22%20y%3D%2260%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%2260%22%20y%3D%2210%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2210%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%201%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%22136%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2212%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%202%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22180%22%20y%3D%22230%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2213%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%202%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22350%22%20y%3D%22230%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2214%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%202%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%22180%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2215%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%203%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22180%22%20y%3D%22270%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2216%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%204%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22350%22%20y%3D%22270%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2217%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bid%20%3D%201%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%2290%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2222%22%20style%3D%22edgeStyle%3DorthogonalEdgeStyle%3Brounded%3D0%3Bhtml%3D1%3BexitX%3D0.5%3BexitY%3D0%3BentryX%3D0.5%3BentryY%3D1%3BjettySize%3Dauto%3BorthogonalLoop%3D1%3BfontSize%3D13%3B%22%20edge%3D%221%22%20parent%3D%221%22%20source%3D%2218%22%20target%3D%2215%22%3E%3CmxGeometry%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CArray%20as%3D%22points%22%3E%3CmxPoint%20x%3D%22310%22%20y%3D%2225%22%2F%3E%3CmxPoint%20x%3D%22130%22%20y%3D%2225%22%2F%3E%3CmxPoint%20x%3D%22130%22%20y%3D%22310%22%2F%3E%3CmxPoint%20x%3D%22220%22%20y%3D%22310%22%2F%3E%3C%2FArray%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2218%22%20value%3D%22%26lt%3Bfont%20style%3D%26quot%3Bfont-size%3A%208px%26quot%3B%26gt%3Bparent%20id%20%3D%203%26lt%3B%2Ffont%26gt%3B%22%20style%3D%22text%3Bhtml%3D1%3BstrokeColor%3Dnone%3BfillColor%3Dnone%3Balign%3Dcenter%3BverticalAlign%3Dmiddle%3BwhiteSpace%3Dwrap%3Boverflow%3Dhidden%3B%22%20parent%3D%221%22%20vertex%3D%221%22%3E%3CmxGeometry%20x%3D%22270%22%20y%3D%2245%22%20width%3D%2280%22%20height%3D%2220%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3C%2Froot%3E%3C%2FmxGraphModel%3E" ondblclick="(function(svg){if(svg.wnd!=null&amp;&amp;!svg.wnd.closed){svg.wnd.focus();}else{var r=function(evt){if(evt.data=='ready'&amp;&amp;evt.source==svg.wnd){svg.wnd.postMessage(decodeURIComponent(svg.getAttribute('content')),'*');window.removeEventListener('message',r);}};window.addEventListener('message',r);svg.wnd=window.open('https://www.draw.io/?client=1&amp;chrome=0&amp;edit=_blank');}})(this);" viewBox="0 0 336 309" style="max-width:100%;max-height:309px;"><defs><filter id="dropShadow"><feGaussianBlur in="SourceAlpha" stdDeviation="1.7" result="blur"/><feOffset in="blur" dx="3" dy="3" result="offsetBlur"/><feFlood flood-color="#3D4574" flood-opacity="0.4" result="offsetColor"/><feComposite in="offsetColor" in2="offsetBlur" operator="in" result="offsetBlur"/><feBlend in="SourceGraphic" in2="offsetBlur"/></filter></defs><g transform="translate(0.5,0.5)" filter="url(#dropShadow)"><rect x="128" y="32" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(160.5,52.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="54" height="19" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 18px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 56px; white-space: nowrap; font-weight: bold; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;background-color:#ffffff;">Shoes</div></div></foreignObject><text x="27" y="19" fill="#000000" text-anchor="middle" font-size="18px" font-family="Helvetica" font-weight="bold">Shoes</text></switch></g><rect x="128" y="123" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(163.5,144.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="48" height="16" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 48px; white-space: nowrap; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><b><font style="font-size: 15px">Sports</font></b></div></div></foreignObject><text x="24" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><rect x="38" y="213" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(72.5,236.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="50" height="13" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 13px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 50px; white-space: nowrap; font-weight: bold; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;">Nike Air</div></div></foreignObject><text x="25" y="13" fill="#000000" text-anchor="middle" font-size="13px" font-family="Helvetica" font-weight="bold">Nike Air</text></switch></g><path d="M 98 213 L 181.96 185.01" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 186.94 183.35 L 181.41 188.89 L 181.96 185.01 L 179.19 182.25 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 188 123 L 188 98.37" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 188 93.12 L 191.5 100.12 L 188 98.37 L 184.5 100.12 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><rect x="208" y="213" width="120" height="60" rx="9" ry="9" fill="#ffffff" stroke="#000000" pointer-events="none"/><g transform="translate(223.5,236.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="88" height="13" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 13px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 88px; white-space: nowrap; font-weight: bold; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;">Rebook Pump</div></div></foreignObject><text x="44" y="13" fill="#000000" text-anchor="middle" font-size="13px" font-family="Helvetica" font-weight="bold">Rebook Pump</text></switch></g><path d="M 268 213 L 200.89 185.42" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 196.03 183.42 L 203.84 182.85 L 200.89 185.42 L 201.18 189.32 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><g transform="translate(165.5,122.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="45" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 45px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = 1</font></div></div></foreignObject><text x="23" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(75.5,216.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="45" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 45px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = 2</font></div></div></foreignObject><text x="23" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(245.5,216.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="45" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 45px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = 2</font></div></div></foreignObject><text x="23" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(178.5,166.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 2</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(88.5,256.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 3</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(258.5,256.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 4</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(178.5,76.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="19" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 21px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">id = 1</font></div></div></foreignObject><text x="10" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><path d="M 188 28 L 188 8 L 8 8 L 8 293 L 98 293 L 98 279.37" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 98 274.12 L 101.5 281.12 L 98 279.37 L 94.5 281.12 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><g transform="translate(165.5,31.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="45" height="12" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; overflow: hidden; max-height: 20px; max-width: 80px; width: 45px; white-space: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;"><font style="font-size: 8px">parent id = 3</font></div></div></foreignObject><text x="23" y="12" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g></g></svg>

This is where it'd help for us too...

## Understand the ``UNION`` statement

As already covered, the ``UNION`` statement basically tells Postgres what to do with duplicate rows. ``UNION ALL`` is basically saying: don't discard dupes eg union all the things, where ``UNION`` discards dupes (if you are familiar with set theory at all, then you are probably familiar with [union](https://en.wikipedia.org/wiki/Union_(set_theory)) already).

So let's try creating the messed up data relationship, by creating an infinite recursive loop.

```
UPDATE category SET parent_id = 3 WHERE id = 1;
```

Then run the same query from the last section:

```
WITH RECURSIVE category_tree AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 3
    UNION ALL
    SELECT category.id, category.name, category.parent_id
        FROM category
        JOIN category_tree ON (category.id = category_tree.parent_id)
)
SELECT * FROM category_tree;
```

If you're playing along at home, you'll notice that query never completes and you'll need to Ctrl + c your way of it. That's because each time the recursive part of the query is run, rows are returned. So it never reaches an end condition.

We can combat this problem by using ``UNION`` instead of ``UNION ALL`` to discards duplicate rows:

```
WITH RECURSIVE top_category AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 3
    UNION
    SELECT category.id, category.name, category.parent_id
        FROM category
        JOIN top_category ON (top_category.parent_id = category.id)
)
SELECT * FROM top_category;
```

Now when the query goes to join the ``Shoes`` row with ``Nike Air``, it finds a duplicate row, since we are discarding dupe, there's nothing left to join so we return. Voilà:

```
 id |     name     | parent_id
----+--------------+-----------
  3 | Rebook Pumps |         2
  2 | Sports       |         1
  1 | Shoes        |         3
(3 rows)
```

Now you may or may not be asking, how does it actually work under the hood? If you're in the former camp, read on...

### Understand (very loosely) how it works

Let's try to understand what's going on here, according to the docs. This isn't me deep diving into the Postgres code base, so it could very well be wrong, but hopefully it's a close enough approximation.

We have already determined that the first step in evaluating recursive queries is to deal with the non-recursive term. According to the docs, the output of this query is place onto a table called the "working table". This output is also appending to our results. So we have 2 data structures that looks like this:

```
Results                         Working table
-----------------------------   --------------------------------
                              
3   Reebok Pumps   2            3   Reebok Pumps   2
```

Next, we evaluate the recursive term against the working table, which in this case is ``SELECT category.id, category.name, category.parent_id FROM category JOIN top_category ON (category.id = top_category.parent_id)``. Effectively we're saying "get a thing from ``category`` which can be joined against our working table". So, we fetch the sports category. This time we add the results to a table called the "intermediate table". Okay, so now we have 3 structure that look like this:

```
Results                         Working table                     Intermediate table
-----------------------------   --------------------------------  --------------------
                              
3   Reebok Pumps   2            3   Reebok Pumps   2              2   Sports   1
```

Then the contents of the intermediate table replace the working table and are appended to our results. So our tables look like this:

```
Results                         Working table                     Intermediate table
-----------------------------   --------------------------------  --------------------
                              
3   Reebok Pumps   2            2   Sports     1                    
2   Sports         1
```

Then we keep following this process until there is nothing to replace into our working table.
