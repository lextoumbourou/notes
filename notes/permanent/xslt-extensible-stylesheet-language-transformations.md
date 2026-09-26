---
title: XSLT (eXtensible Stylesheet Language Transformations)
date: 2025-08-20 00:00
modified: 2025-08-20 00:00
status: draft
---

XSLT is a template-based transformation language that processes [XML](xml.md) documents by walking through the tree structure and applying pattern-matching rules to convert XML into various output formats like HTML, CSV, or other XML. It works through a pipeline system where templates define "when you encounter this pattern, produce this output," allowing complex transformations to be built from simple, reusable template rules.

## Example

### Input XML (albums.xml)

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

### XSLT Stylesheet (albums.xsl)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:template match="/">
        <html>
            <head>
                <title>90s Alternative Rock Collection</title>
            </head>
            <body>
                <h1>Classic Grunge Albums</h1>
                <xsl:for-each select="musicCollection/album">
                    <div class="album">
                        <xsl:if test="genre='Grunge'">
                            <h2><xsl:value-of select="title"/></h2>
                            <p><strong>Artist:</strong> <xsl:value-of select="artist"/></p>
                            <p><strong>Released:</strong> <xsl:value-of select="year"/></p>
                        </xsl:if>
                    </div>
                </xsl:for-each>
            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>
```

### Run command using xsltproc:

```
xsltproc albums.xsl albums.xml
```

### Output

```html
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>90s Alternative Rock Collection</title>
</head>
<body>
<h1>Classic Grunge Albums</h1>
<div class="album">
<h2>Nevermind</h2>
<p><strong>Artist:</strong>Nirvana</p>
<p><strong>Released:</strong>1991</p>
</div>
<div class="album"></div>
</body>
</html>
```

