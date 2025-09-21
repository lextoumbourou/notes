---
title: XPath
date: 2025-09-21 00:00
modified: 2025-09-21 00:00
status: draft
---

**XPath** is a query language for [XML](../../../permanent/xml.md) that kinda resembles using a file-system. It was another standard defined by the W3C, and is commonly used in web scraping frameworks.

Imagine you had this XML document, a list of classic albums:

```xml
<albums>
    <album id="1" genre="grunge">
        <title>Nevermind</title>
        <artist>Nirvana</artist>
        <year>1991</year>
        <label>DGC Records</label>
    </album>
    <album id="2" genre="hip-hop">
        <title>My Beautiful Dark Twisted Fantasy</title>
        <artist>Kanye West</artist>
        <year>2010</year>
        <label>Roc-A-Fella Records</label>
    </album>
</albums>
```

The most basic form of XPath queries is the selectors. If you've used a terminal like bash, a lot of these operators are familiar to you.

| Operator | Description                            | Example                                                    |
| -------- | -------------------------------------- | ---------------------------------------------------------- |
| `/`      | Select from root node                  | `/albums/album` - returns a list of albums<br>             |
| `//`     | Select nodes anywhere in the document. | `//year` - return all album years                          |
| `@`      | Select attributes                      | `//album[@genre='grunge']/artist` - get all grunge artists |
| `.`      | Current node                           |                                                            |
| `..`     | Parent node                            |                                                            |

All of these queries return a set of results, but you can index into the results using familiar array notation (XPath is 1-based indexed):

```xpath
//albums/album[1] # Get first result
```

```xpath
//albums/album[last()] # Get last result
```

You can do position-based, attribute-based and content-based filtering.

For example, Position-based filtering:

```xpath
//album[2]                    # Second album
//album[position() > 1]       # All albums except the first
//album[last()-1]             # Second-to-last album
```

Attribute-based filtering:

```xpath
//album[@genre]               # Albums that have a genre attribute
//album[@genre='grunge']      # Albums with genre="grunge"
//album[@id > 1]              # Albums with id greater than 1
```

Content-based filtering:

```xpath
//album[year > 2000]          # Albums released after 2000
//album[artist='Nirvana']     # Albums by Nirvana
//album[contains(title, 'Dark')] # Albums with "Dark" in the title
```

XPath also includes many built-in functions that make querying more powerful:

String functions:
- `contains(string, substring)` - Check if string contains substring
- `starts-with(string, prefix)` - Check if string starts with prefix
- `normalize-space(string)` - Remove leading/trailing whitespace
- `string-length(string)` - Get length of string

Numeric functions:
- `count(nodeset)` - Count the number of nodes
- `sum(nodeset)` - Sum numeric values
- `floor()`, `ceiling()`, `round()` - Math operations

Node functions:
- `position()` - Current node position
- `last()` - Position of last node
- `name()` - Get node name
- `text()` - Get text content

You can combine multiple conditions using logical operators:

```xpath
//album[@genre='grunge' and year < 1995]  # Grunge albums before 1995
//album[year < 1995 or year > 2005]       # Albums from before 1995 or after 2005
//album[not(@genre='pop')]                # Albums that aren't pop genre
```

Beyond basic selectors, XPath provides "axes" that let you navigate relationships between nodes:

```xpath
//artist/following-sibling::year    # Get year that follows artist
//album/descendant::text()          # All text content within album
//title/ancestor::album             # Get album that contains this title
//album/child::*                    # All direct children of album
```

XPath is particularly popular in web scraping because it can navigate HTML's DOM structure just like XML. Most web scraping libraries support XPath selectors:

```python
# Using Selenium (current syntax)
from selenium.webdriver.common.by import By
driver.find_element(By.XPATH, "//div[@class='product-price']")

# Using lxml
tree.xpath("//a[contains(@href, 'product')]/@href")

# Using Scrapy
response.xpath("//h1[@class='title']/text()").get()
```