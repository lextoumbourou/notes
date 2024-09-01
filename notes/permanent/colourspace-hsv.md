---
title: Hue Saturation Value (HSV)
date: 2024-09-01 00:00
modified: 2024-09-01 00:00
status: draft
---

**HSV** or **Hue Saturation Value** is a [Colour Space](colour-space.md) alternate to [RGB](rgb.md).

The HSV colour space corresponds better to how people experience colour than the RGB colour space does. For example, this colour space is often used by people who select colours, such as paint or ink, from a colour wheel or palette.

The Hue (H) corresponds to the colour's position on a colour wheel and is $[0, 1]$. As H increases, the colours transition from red to orange, yellow, green, cyan, blue, magenta, and finally back to red. Both 0 and 1 indicate red.

Saturation (S) is the amount of hue or departure from neutral. S is in the range $[0, 1]$. As S increases, colours vary from unsaturated (shades of grey) to fully saturated (no white component).

Value (V) is the maximum value among a specific colour's red, green, and blue components. V is in the range $[0, 1]$. As V increases, the corresponding colours become increasingly brighter.

The HSV colour space is an inverted cone, where hue relates to the angle, saturation to the radius, and value to the height from the origin. White is at the origin, and white is at the furthest point along the value axis.

![Diagram of HSV Colour Space by Mathworks](../_media/HSV-img.png)

Image from [Understanding Color Spaces and Color Space Conversion](https://au.mathworks.com/help/images/understanding-color-spaces-and-color-space-conversion.html) by [Mathworks](https://au.mathworks.com/).