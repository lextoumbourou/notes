---
title: "Week 8: Applications of Sine and Cosine Rules"
date: 2023-06-10 00:00
modified: 2023-06-10 00:00
status: draft
---

* Application of Sine and Cosine Rules
    * Some examples that utilise sine and cosine rule.
* Triangle Rectangle Isosceles
    * We have a right-triangle with hypothenuse $h$ and 2 other sides that have the same length.
        ![](../../../../journal/_media/week-8-applications-of-sine-and-cosine-rules-right-triangle.png)
    * Since it's an [[Isosceles Triangle]], meaning on angle is 90°, we know that the 2 angles adjacent to h are equal, which means they're 45°.
    * Pythagoras theorem gives us $a^2 + b^2 = 2a^2 = h^2$
        * That means, $2a^2 = h^2$
        * Which means $a = h / \sqrt(2)$
        * We know that the ratio of adjacent to hypotenuse is $sin(90 - \theta)$ so $a/h=1/\sqrt(2) = \sin(45°)$
        * $\sin(45°) = 1/\sqrt(2) = \sqrt(2) / 2$
* Finding the length of right-hand side of larger triangle with [[../../../../permanent/similar-triangles]]
    * ![](../../../../journal/_media/week-8-applications-of-sine-and-cosine-rules-similar-tri.png)
        * We know that 1cm / X = 2.5 / 5cm
        * So X = 2cm.
* Generic triangle
    * $a = 8cm, b = 3cm, \alpha = 58°, c?$
        * Note: $\sin(58°) = 0.848$
         ![](../../../../journal/_media/week-8-applications-of-sine-and-cosine-rules-generic-triangle.png)
    * Use sine ratio: $a / \sin(\alpha)  = b / \sin(\beta) = c / \sin(\gamma)$
        * $\sin(\beta) = b \sin(\alpha) / a \rightarrow \sin(\beta) = 3 \times 0.848 / 8 = 0.318$
    * We can evaluate $\beta$ by inverting the $\sin$
        * $\beta = Sin^{-1}(0.318) = 18.54°$
    * Can evaluate $\gamma$ as we know it's 180 - sum of other angles.
        * $\gamma = 180° - 18.54° - 58° = 103.46°$
    * We know $c/ \sin(\gamma) = a / \sin(\alpha)$
    * $c = a \sin(\gamma) / \sin(\alpha) = 8 \times 0.97 / 0.848 = 9.15cm$
* Generic triangle example
        ![](../../../../journal/_media/week-8-applications-of-sine-and-cosine-rules-generic-triangle-1.png)
    * $a = 6cm$, $b = 4cm, $c = 3cm, $\alpha = ?$
    * Use cosine rule: $a^2 = b^2 + c^2 -2bc \cos(\alpha)$
    * $\cos(\alpha) = \frac{b^2 + c^2 - a^2}{2bc} = {16 + 9  -36}{24} = -0.458$
    * $\alpha = cos^{-1}(-0.458) = 117.3°$
