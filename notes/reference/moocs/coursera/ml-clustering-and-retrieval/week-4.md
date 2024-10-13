---
title: Mixture Models
date: 2021-10-30 00:00
modified: 2023-04-08 00:00
status: draft
---

# Mixture Models

## Motivating probabilistic clustering models

* Why a probabilistic approach?
    * Rarely in situation with clear cut cluster differences which [K-Means](../../../../permanent/k-means.md) implies.
    * Often have overlapping clusters, yet k-means does hard assignment.
    * Also, cluster centers may not be most important point: may want to take into account shape of cluster.
* [Mixture Model](../../../../../../permanent/mixture-model.md)
      * Can provide "soft assignment" of observations to clusters.
    * Example: 54% fashion, 30% beauty, 16% travel etc.
    * Can account for cluster shape, not just focused on cluster center.

## Aggregating over unknown classes in an image dataset

* Basic idea: clustering with multiple dimensions can uncover groups that a single dimension may struggle with.
    * Example: using RBG values to detect groups of images. If only using green to classify, may be difficult to distinguish forest and sunset until red is introduced as dimension.

## Univariate Gaussian distributions

* Gaussian distribution
    * Aka normal distribution.
    * Fully specified by **mean** μ and **variance** σ^2 (or **standard deviation** σ).
    * Notated as follows:

        $$N(x | \mu, \sigma^2) $$

    * x represents the random variable the distribution is over (eg blue intensity)
    * μ and σ^2 represent the fixed "parameters" of the gaussian.

## Bivariate and multivariate gaussians

* Gaussian in 2 or more dimensions aka distribution of multiple random variables.
    * Examples: word count of "I" and "we" in social media bios to determine person or people.
* When 2 dimensions is drawn as a 3d mesh plot:
    * Most probable region at tip of peak.
    * Least probable towards the bottom.
* Contour plot:
    * "Birds eye view" of 3d mesh plot.
* In 2D: gaussian fully specified by **mean** ***vector*** and **covariance** ***matrix***
* Mean vector:

    $$\mathbf{\mu} = [\mu_{\color{blue}{\text{blue}}}, \mu_{\color{green}{\text{green}}}] $$

   * Mean "centres the distribution in 2D"
       * The middle point between the 2 values means is the centre point of the contour plot.
* Covariance matrix:

 $$\Sigma = \begin{bmatrix}{\sigma_\color{blue}{blue}}^2 & \sigma_{\color{blue}{blue},\color{green}{green}} \\ \sigma_{\color{green}{green},\color{blue}{blue}} & {\sigma_\color{green}{green}}^2 \end{bmatrix}$$

    * determines orientation and spread aka correlation structure (don't quite get this yet).
* Covariance structure examples:
    * Diagonal covariance with equal elements over the diagonal
     * $$\Sigma = \begin{bmatrix}\sigma^2 & 0 \\ 0 & \sigma^2 \end{bmatrix} $$
     * "There's no correlation between the random variables"
     * Get a circulate shape to the distribution because variables on both dimensions are the same.
   * Diagonal covariance with different variances over the diagonal
     * $$\Sigma = \begin{bmatrix}{\sigma_\color{blue}{B}}^{2} & 0 \\ 0 & {\sigma_\color{green}{G}}^2 \end{bmatrix} $$
     * End up with axis aligned ellipses.
   * Full covariance allowing for correlation between variables.
     * $$\Sigma = \begin{bmatrix}{\sigma_\color{blue}{B}}^{2} & \sigma_{\color{blue}{B},\color{green}{G}} \\ \sigma_{\color{green}{G},\color{blue}{B}} & {\sigma_\color{green}{G}}^2 \end{bmatrix} $$
     * End up with non axis aligned axis.
* Projects to help me understand this:
  * Write function to return 2 random variables with different mean vectors and covariance matrixes and build histogram of values.
* Notation to describe Guassian: $$N(\mathbf{x} \mid \mathbf{\mu}, \mathbf{\Sigma}) $$

## Mixtures of Gaussians for clustering

### Mixture of Gaussians

* Special case of mixture model: [Gaussian Mixture Model](../../../../../../permanent/gaussian-mixture-model.md)
* Each class of images has a multi variable gaussian distribution of the different colour values.
* Since we initially don't know the image classes, we just have a bunch of gaussians over the entire dataset space.
* Question: how are you going to model the colour distribution across the entire dataset?
    * Taking the category specific gaussians (but we don't know the labels yet???) and average them together.
* Utilise a weighted average to account for larger numbers of datapoints for a certain category.
  * Introduce cluster weights $$\pi_k $$

where k represents a cluster of datapoints.
  * Each weight is between 0 and 1.
  * Sum of the weights always equals or integrates to 1.
  * Called a "convex combination".
  * Example: $$\mathbf{\pi} = [\stackrel{\pi_1}{0.47}, \stackrel{\pi_2}{0.26}, \stackrel{\pi_3}{0.27}] $$
  * Each mixture component represents a unique cluster specified by $$\{\pi_k, \mathbf{\mu_k}, \Sigma_k\} $$

### Interpreting the mixture of Gaussian terms

* Prior term:
    * $$p(z_i = k) = \pi_k $$
    * Aka prior probability: if you have don't know what the datapoint is, how likely is it?
* Likelihood term:
    * Given $$\mathbf{x}_i $$is from cluster k, what's the likelihood of seeing $$\mathbf{x}_i $$

?

    * $$p(x_i | z_i = j, \mu_k, \Sigma_k) = N(x_i \mid \mu_k, \Sigma_k) $$

### Scaling mixtures of Gaussians for document clustering

* Allowing for a full covariance matrix would require over V(V + 1) / 2 unique parameters to learn.
* Assume diagonal form to covariance matrix.
  * Means ellipses are constrained to axis aligned.
  * Somewhat restrictive but mitigated by the fact you can learn weights of dimensions and can even learn them within clusters: may discover, for example, one word is more important in one cluster than another.

## Expectation Maximization (EM) building blocks

### Computing soft assignments from known cluster parameters

* Finally here: algorithm to infer cluster parameters is called "Expectation Maximisation" aka EM.
* Steps to infer soft assignments with expectation maximisation (EM):
  1. Start with unlabelled observed inputs.
  2. Output soft assignments per data point.

* Part 1: What if we know the cluster params $$\{\pi_k, \mu_k, \Sigma_k\} $$

?
    * Easy to compute when known: for each possible cluster, compute prior by likelihood to get responsibily and normalise vector so it sums to 1 over all clusters.
    * Soft assignments are identified by a thing called "responsibility vector":

      $$r_i = [r_i1, r_i2 ... r_ik] $$

      (where k is the number of clusters in the model)

    * $$r_{ik} = \frac{\pi_k \space N(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \space N(x_i \mid \mu_k, \Sigma_k)} $$

  * When datapoint is closer to cluster centre, cluster is said to take "more responsibility" for datapoint.

### Estimating cluster parameters from known (hard) cluster assignments

* Part 2: Imagine we knew the hard cluster assignments (eg datapoint is either in a cluster of isn't) and want to infer cluster params.
    * Mean: average of all data points in the cluster: $$1/N_k \sum\limits_{\text{i in k}} x_i $$
    * Covariance: average of the distance from the mean for all data points: $$1/N_k \sum\limits_{\text{i in k}} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T $$
    * Cluster proportions: observations in cluster / total observations: $$\hat{\pi}_k = \frac{N_k}{N} $$

### Estimating cluster parameters from soft assignments

* Part 2b: Imagine we knew the soft cluster assignments and want to infer cluster params.
  * Same calculations as previous section but weighing each datapoint by their cluster assignment weights.

## The EM algorithm

### EM iterates in equations and pictures

1. First step is to initialise iter counter.

    $$\{{\pi_k}^{(0)}, \hat{\mu_k}^{(0)}, \hat{\Sigma_k}^{(0)}\} $$

2. Next, estimate cluster responsibilities given current parameter estimations (E-step)

     $$r_{ik} = \frac{\pi_k \space N(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \space N(x_i \mid \mu_k, \Sigma_k)} $$

3. Last, maximise likelihood over parameters given calculated responsibilities (M-step)

### Convergence, initialization and overfitting of EM

* EM is a coordinate-ascent algorithm
  * E and M steps same as alternating maximisations of an objective function.
* Converges to a local mode.
* Initialisation:
  * Many way to init.
  * Important for convergence rates and quality of local mode.
  * Examples:
    * Choose observations at random to be centres.
    * Choose centres sequentially ala k-means++.
    * Run K-means and use those centres.
* Overfitting of MLE:
  * Maximising likelihood can overfit data.
  * A cluster that has only one value assigned to it would have an infinite likelihood function which would dominate overall likelihood function.
* Overfitting high dimensions:
  * Imagine only 1 doc assigned to cluster k has word w (or all docs in cluster agree on count of word w).
  * Likelihood of any other doc with different count on word w being in cluster k is 0.
* Simple fix: don't let variances get to 0. Add small amount to diagonal of covariance estimate.
  * Alternatively: take Bayesian approach and place prior's on parameters.

### Relationship to k-means

* Summary: gaussian mixture models becomes near identical to k-means when variance parameter is set to 0.
* If you consider a gaussian mixture model with spherically symmetric clusters (sigma squared is identical along covariance matrix diagonal) and set variance to 0:
  * Clusters have equal variance so relative likelihood is just function of cluster centre.
  * As variance goes to 0, likelihood ratio becomes 0 or 1.
  * Responsibilities way in somewhat, but dominated by likelihood.

### Worked example for EM

#### E-step: compute cluster resposbilities given cluster params

Use whatever info we have at start time. If none, assign cluster centres randomly, and pick a diagonal covariance, then pick weights that give each cluster an even spread (1/3).

##### Datapoint 0

```
In [4]: print 1/3. * multivariate_normal.pdf([10, 5], mean=[3, 4], cov=[3, 0](3,%200))
4.2506655934e-06

In [5]: print 1/3. * multivariate_normal.pdf([10, 5], mean=[6, 3], cov=[3, 0](3,%200))
0.000630854709005

In [6]: print 1/3. * multivariate_normal.pdf([10, 5], mean=[4, 6], cov=[3, 0](3,%200))
3.71046481027e-05

In [7]: 4.2506655934e-06 + 0.000630854709005 + 3.71046481027e-05
Out[7]: 0.0006722100227011
```

             Cluster A            | Cluster B          | Cluster C          | Sum

Likelihood 4.2506655934e-06 | 0.000630854709005 | 3.71046481027e-05 | 0.0006722100227011
L / sum 0.007 | 0.938 | 0.055

##### Datapoint 1

```
In [11]: print 1/3. * multivariate_normal.pdf([2, 1], mean=[3, 4], cov=[3, 0](3,%200))
0.00334005398012

In [12]: print 1/3. * multivariate_normal.pdf([2, 1], mean=[6, 3], cov=[3, 0](3,%200))
0.000630854709005

In [13]: print 1/3. * multivariate_normal.pdf([2, 1], mean=[4, 6], cov=[3, 0](3,%200))
0.000140762712251

In [14]: 0.00334005398012 + 0.000630854709005 + 0.000140762712251
Out[14]: 0.004111671401376

In [15]: 0.00334005398012 / 0.004111671401376
Out[15]: 0.8123348521971447

In [16]: 0.000630854709005 / 0.004111671401376
Out[16]: 0.15343023491465782

In [17]: 0.000140762712251 / 0.004111671401376
Out[17]: 0.034234912888197425
```

             Cluster A           | Cluster B           | Cluster C          | Sum

Likelihood 0.00334005398012 | 0.000630854709005 | 0.000140762712251 | 0.004111671401376
L / sum 0.8123348521971447 | 0.15343023491465782 | 0.034234912888197

##### Datapoint 2

```
In [18]: print 1/3. * multivariate_normal.pdf([3, 7], mean=[3, 4], cov=[3, 0](3,%200))
0.00394580754895

In [19]: print 1/3. * multivariate_normal.pdf([3, 7], mean=[6, 3], cov=[3, 0](3,%200))
0.000274168326362

In [20]: print 1/3. * multivariate_normal.pdf([3, 7], mean=[4, 6], cov=[3, 0](3,%200))
0.0126710555509

In [21]: 0.00394580754895 + 0.000274168326362 + 0.0126710555509
Out[21]: 0.016891031426212

In [22]: 0.00394580754895 / 0.016891031426212
Out[22]: 0.23360370656979415

In [23]: 0.000274168326362 / 0.016891031426212
Out[23]: 0.016231591750906195

In [24]: 0.0126710555509 / 0.016891031426212
Out[24]: 0.7501647016792996
```

             Cluster A           | Cluster B           | Cluster C          | Sum

Likelihood 0.00394580754895 | 0.000274168326362 | 0.0126710555509 | 0.016891031426212
L / sum 0.23360370656979415 | 0.01623159175090619 | 0.7501647016792996

##### Full responsibility matrix

```
                 Cluster A | Cluster B | Cluster C
Data point 0  |  0.007     | 0.938     | 0.055
Data point 1  |  0.812     | 0.153     | 0.034
Data point 2  |  0.234     | 0.016     | 0.75
Soft counts   |  1.053     | 1.108     | 0.839
```

#### M-step: compute cluster parameters, given cluster responsibilities

Get cluster weights by adding up soft counts and using to normalise

```
                 Cluster A | Cluster B | Cluster C | Sum
Soft counts   |  1.053     | 1.108     | 0.839     | 1.053 + 1.108 + 0.839 = 3
Normalized    |  0.351     | 0.369     | 0.280
```

##### Mean

Get means by adding "fractional parts" of all data points using cluster responsibilities

```
= 0.007 * (10, 5) + 0.812 * (2, 1) + 0.234 * (3, 7)
= (0.07, 0.035) + (1.624, 0.812) + (0.702, 1.638)
= (2.396, 2.485)
```

Then divide sum by the soft count.

```
= (2.396, 2.485) / 1.053 
= (2.275, 2.36)
```

Then repeat for other clusters to get new mean estimates

```
New means    | X        | Y

Cluster A    | 2.275    | 2.360
Cluster B    | 8.787    | 4.473
Cluster C    | 3.418    | 6.626
```

##### Covariance

Compute difference from the mean like so: ``data point i - cluster mean k``

```
# point 0
= (10, 5) - (2.275, 2.360) = (7.725, 2.640)

# point 1
= (2, 1) - (2.275,2.360) = (-0.275, -1.360)

# point 2
= (3, 7) - (2.275, 2.360) = (0.725, 4.640)
```

Then compute the "outer products" which are two-by-two matrices:

```
= [7.725](7.725) * [7.725,2.640](7.725,2.640) = [59.676,20.394](59.676,20.394)

= [-0.275](-0.275) * [-0.275,-1.360](-0.275,-1.360) = [0.076,0.374](0.076,0.374)

= [0.725](0.725) * [0.725,4.640](0.725,4.640) = [0.526,3.364](0.526,3.364)
```

Then take the weighted average using cluster responsibilities:

```
=  0.007 * [59.676,20.394](59.676,20.394) + 0.812*[0.076,0.374](0.076,0.374) + 0.234*[0.526,3.364](0.526,3.364)
= (0.602, 1.234), (1.234, 6.589)
```

Normalise with soft count (??)

```
= ((0.602, 1.234), (1.234, 6.589)) / 1.053
= ((0.572, 1.172), (1.172, 6.257))
```

##### Alternating between E-step and M-step

Then, you can keep alternating between the E-step and M-step and each time they will improve the quality of each other.
