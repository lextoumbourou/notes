---
title: Rotating image by any angle
date: 2021-10-28 00:00
category: reference/articles
status: draft
---

This article contains notes from blog post [Rotating Image By Any Angle(Shear Transformation) Using Only NumPy](https://gautamnagrawal.medium.com/rotating-image-by-any-angle-shear-transformation-using-only-numpy-d28d16eb5076) by Gautam Agrawal.

Each pixel in an image has a coordinate pair (x, y) that describes its position.

We can obtain the new location by multiplying by the following transformation matrix:

$\begin{bmatrix}\cos\theta && \sin\theta \\ -\sin\theta && \cos\theta \end{bmatrix}$

To understand what's happening with this transformation matrix, when the angle of rotation is 0, we get the following results:

$\cos(0°) = 1$
$\sin(0°) = 0$

So the [Matrix Transformation](../../permanent/matrix-transformation.md) simply contains the basis vectors:

$\begin{bmatrix}1 && 0 \\ 0 && 1\end{bmatrix}$

On the other hand, when the rotation is a 90° rotation, we get the following results:

$\cos(90°) = 0$
$\sin(90°) = 1$

So our transformation matrix looks like this:

$\begin{bmatrix}0 && 1 \\ -1 && 0\end{bmatrix}$

We use this formula to find the dimensions of the new image:

$\text{new width} = \| \text{old width} \times \cos\theta \| + |\text{old height} \times \sin\theta \|$
$\text{new height} = \| \text{old height} \times \cos\theta \| + \| \text{old width} \times \sin\theta \|$

Again, in the two extreme examples above, we can see how $cos(0°)$ would result in the same height and width, and $cos(90°)$ would result in inversing the height and width.

The last part ensures that we define coordinate vectors from the center point, not the top-left. Otherwise, if we did a 90° rotation, the entire image would end up out of bounds.

Because multiplying integer coordinates results in fractional values, when we round them back to ints, this means some spots get addressed more than once, and other pixels get missed entirely. The closer the angle is to a diagonal, the worse it gets.

One solution is to oversample the source image: pretend each source pixels are $n \ \times \ n$ grids of smaller pixels, and calculate coordinates of subpixels.

Another approach is Area Mapping. Calculate the color of each destination pixel by a weighted average of four source pixels.

Yet another approach is the three shear rotation.

Turn the rotation into 3 separate sheer operations:

$\begin{bmatrix}1 && -\tan(\theta/2) \\ 0 && 1\end{bmatrix} \begin{bmatrix}1 && 1 \\ \sin\theta && 1\end{bmatrix} \begin{bmatrix}1 && -\tan(\theta/2) \\ 0 && 1\end{bmatrix}$

1. The three matrices are all shear matrices.
2. The first and last matrices are the same.
3. [Matrix Determinate](../../permanent/matrix-determinate.md) of each matrix is the same.
4. Since the shear happens in just one plane, and each stage is *conformal* in the area, no aliasing gaps appear.
