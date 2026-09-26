---
title: Ridge Regression
date: 2024-10-12 00:00
modified: 2024-10-12 00:00
status: draft
---

**Ridge Regression** is a [Linear Regression](linear-regression.md) variant that adds an [L2 Norm](l2-norm.md) penalty term (the sum of the squares of the coefficients) to the cost function. This regularisation helps prevent overfitting by shrinking the model's coefficients towards zero. It is especially useful when the data has multicollinearity or when you want to regularise large coefficients without eliminating them completely.

An alternative to Ridge Regression is [Lasso](lasso.md), which uses an [L1 Norm](l1-norm.md) penalty, constraining the sum of the absolute values of the coefficients. Lasso can shrink some coefficients to zero, making it suitable for feature selection.