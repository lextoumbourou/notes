---
title: CIE Lab
date: 2024-09-01 00:00
modified: 2026-09-26 08:55
status: draft
---

CIE $L^{*}a^{*}b^{*}$ colour space, which is derived from the [CIE XYZ (1931)](cie-xyz-1931.md) color space and is designed to linearize the perceptibility of color differences, though it remains non-linear. The L\* component, representing lightness, is calculated using different formulas depending on the value of $\frac{Y}{Y_n}$, where $Y$ is the luminance and $Y_n$ is the reference white luminance. The a\* and b\* components represent chromaticity and are computed from non-linear functions of the XYZ coordinates.

The formulas are given as:

$$
L^* = \begin{cases} 
116\left(\frac{Y}{Y_n}\right)^{\frac{1}{3}} - 16 & \text{if } \frac{Y}{Y_n} > 0.008856 \\
903.3 \left(\frac{Y}{Y_n}\right) & \text{if } \frac{Y}{Y_n} \leq 0.008856 
\end{cases}
$$

$$
a^* = 500 \times \left(f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right)\right)
$$

$$
b^* = 200 \times \left(f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right)\right)
$$

where the function $f(t)$ is defined as:

$$
f(t) = \begin{cases} 
t^{\frac{1}{3}} & \text{if } t > 0.008856 \\
7.787t + \frac{16}{116} & \text{if } t \leq 0.008856 
\end{cases}
$$

Additionally, the chroma $C^*$ and hue angle $h_{ab}$ are defined as:

$$
C^* = \sqrt{a^{*2} + b^{*2}}
$$

$$
h_{ab} = \arctan\left(\frac{b^*}{a^*}\right)
$$

Lab is a color space designed to approximate human vision, with L representing lightness and a and b representing color dimensions. It's particularly useful for color management across different devices due to its device-independent nature and perceptual uniformity. Lab allows for precise color communication and is widely used in industries where accurate color reproduction is crucial, such as printing, textiles, and product design.

* L: Lightness (0 to 100)
* a: Green-Red axis (-128 to +127)
* b: Blue-Yellow axis (-128 to +127)

Part of the family of [Colourimetry](colourimetry.md) colour systems, and is based on [CIE XYZ (1931)](cie-xyz-1931.md)