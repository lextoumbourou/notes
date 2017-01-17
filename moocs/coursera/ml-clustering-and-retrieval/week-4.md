# Mixture Models

## Motivating probabilistic clustering models

* Why a probabilistic approach?
  * Rarely in situation with clear cut cluster differences which k-means implies.
  * Often have overlapping clusters, yep k-means does hard assignment.
  * Cluster centers may not be most important point: may want to take into account shape of cluster.

* Mixture model:
  * Can provide "soft assignment" of observations to clusters.
    * Example: 54% fashion, 30% beauty, 16% travel etc.
  * Can account for cluster shape, not just focused on cluster center.

## Aggregating over unknown classes in an image dataset

* Basic idea: clustering with multiple dimensions can uncover groups that a single dimension may struggle with.
  * Example: using RBG values to detect groups of images. If only using green to classify, may be difficult to distinguish forest and sunset until red is introduced as dimension.

## Univariate Gaussian distributions

* Gaussian distribution:
  * Aka normal distribution.
  * Fully specified by **mean** μ and **variance** σ^2 (or **standard deviation** σ).
  * Notated as follows:
    $$ N(x | \mu, \sigma^2) $$
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
   * $$ \mathbf{\mu} = [\mu_{\color{blue}{\text{blue}}}, \mu_{\color{green}{\text{green}}}] $$
   * Mean "centres the distribution in 2D"
       * The middle point between the 2 values means is the centre point of the contour plot.

* Covariance matrix:

   * $$ \Sigma = \begin{bmatrix}{\sigma_\color{blue}{blue}}^2 & \sigma_{\color{blue}{blue},\color{green}{green}} \\ \sigma_{\color{green}{green},\color{blue}{blue}} & {\sigma_\color{green}{green}}^2 \end{bmatrix} $$
   * determines orientation and spread aka correlation structure (don't quite get this yet). 
* Covariance structure examples:
   * Diagonal covariance with equal elements over the diagonal 
     * $$ \Sigma = \begin{bmatrix}\sigma^2 & 0 \\ 0 & \sigma^2 \end{bmatrix} $$ 
     * "There's no correlation between the random variables"
     * Get a circulate shape to the distribution because variables on both dimensions are the same.
   * Diagonal covariance with different variances over the diagonal 
     * $$ \Sigma = \begin{bmatrix}{\sigma_\color{blue}{B}}^{2} & 0 \\ 0 & {\sigma_\color{green}{G}}^2 \end{bmatrix} $$ 
     * End up with axis aligned ellipses.
   * Full covariance allowing for correlation between variables.
     * $$ \Sigma = \begin{bmatrix}{\sigma_\color{blue}{B}}^{2} & \sigma_{\color{blue}{B},\color{green}{G}} \\ \sigma_{\color{green}{G},\color{blue}{B}} & {\sigma_\color{green}{G}}^2 \end{bmatrix} $$ 
     * End up with non axis aligned axis.
* Projects to help me understand this:
  * Write function to return 2 random variables with different mean vectors and covariance matrixes and build histogram of values.
* Notation to describe Guassian: $$ N(\mathbf{x} \mid \mathbf{\mu}, \mathbf{\Sigma}) $$

## Mixtures of Gaussians for clustering

### Mixture of Gaussians

* Special case of mixture model: mixture of gaussians.
* Each class of images has a multi variable gaussian distribution of the different colour values.
* Since we initially don't know the image classes, we just have a bunch of gaussians over the entire dataset space.
* Question: how are you going to model the colour distribution across the entire dataset?
    * Taking the category specific gaussians (but we don't know the labels yet???) and average them together.

* Utilise a weighted average to account for larger numbers of datapoints for a certain category.
  * Introduce cluster weights $$ \pi_k $$ where k represents a cluster of datapoints.
  * Each weight is between 0 and 1.
  * Sum of the weights always equals or integrates to 1.
  * Called a "convex combination".
  * Example: $$ \mathbf{\pi} = [\stackrel{\pi_1}{0.47}, \stackrel{\pi_2}{0.26}, \stackrel{\pi_3}{0.27}] $$
  * Each mixture component represents a unique cluster specified by $$ \{\pi_k, \mathbf{\mu_k}, \Sigma_k\} $$ 

### Interpreting the mixture of Gaussian terms

* Prior term:
    * $$ p(z_i = k) = \pi_k $$
    * Aka prior probability: if you have don't know what the datapoint is, how likely is it?
* Likelihood term:
    * Given $$ \mathbf{x}_i $$ is from cluster k, what's the likelihood of seeing $$ \mathbf{x}_i $$?
    * $$ p(x_i | z_i = j, \mu_k, \Sigma_k) = N(x_i \mid \mu_k, \Sigma_k) $$ 

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

* Part 1: What if we know the cluster params $$ \{\pi_k, \mu_k, \Sigma_k\} $$?
    * Easy to compute when known: for each possible cluster, compute prior by likelihood to get responsibily and normalise vector so it sums to 1 over all clusters.
    * Soft assignments are identified by a thing called "responsibility vector":

      $$ r_i = [r_i1, r_i2 ... r_ik] $$

      (where k is the number of clusters in the model)
    
    * $$ r_{ik} = \frac{\pi_k \space N(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \space N(x_i \mid \mu_k, \Sigma_k)} $$ 

  * When datapoint is closer to cluster centre, cluster is said to take "more responsibility" for datapoint.

### Estimating cluster parameters from known (hard) cluster assignments

* Part 2: Imagine we knew the hard cluster assignments (eg datapoint is either in a cluster of isn't) and want to infer cluster params.
    * Mean: average of all data points in the cluster: $$ 1/N_k \sum\limits_{\text{i in k}} x_i $$
    * Covariance: average of the distance from the mean for all data points: $$ 1/N_k \sum\limits_{\text{i in k}} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T $$
    * Cluster proportions: observations in cluster / total observations: $$ \hat{\pi}_k = \frac{N_k}{N} $$

### Estimating cluster parameters from soft assignments

* Part 2b: Imagine we knew the soft cluster assignments and want to infer cluster params.
  * Same calculations as previous section but weighing each datapoint by their cluster assignment weights.

## The EM algorithm

