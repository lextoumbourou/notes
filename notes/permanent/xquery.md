---
title: XQuery
date: 2025-08-20 00:00
modified: 2026-09-26 08:55
status: draft
---

**XQuery** is a query language for [XML](xml.md) documents.

## Example

Given this XML:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<musicCollection>
    <album>
        <title>Nevermind</title>
        <artist>Nirvana</artist>
        <year>1991</year>
        <genre>Grunge</genre>
    </album>
    <album>
        <title>Illmatic</title>
        <artist>Nas</artist>
        <year>1994</year>
        <genre>Hip-hop</genre>
    </album>
</musicCollection>
```

We can return all albums that match the Grunge genre formatted as HTML as follows:

```xquery
let $albums := doc("books.xml")/musicCollection/album
return
<html>
    <head><title>Grunge Albums</title></head>
    <body>
        <h1>Classic Grunge Albums</h1>
        {
            for $album in $albums
            where $album/genre = 'Grunge'
            order by $album/year ascending
            return
                <div>
                    <h2>{data($album/title)}</h2>
                    <p>Artist: {data($album/artist)}</p>
                    <p>Year: {data($album/year)}</p>
                </div>
        }
    </body>
</html>
```
