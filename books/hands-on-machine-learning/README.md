# Hands On Machine Learning

## Chapter 1

### Exercises

1. How would you define Machine Learning?

The process of learning a model utilising training data which can then be used to make future predictions against new data points.

2. Name 4 types of problems where it shines?

1. Classification problems with no algorithmic solution (aka voice recognition).
2. Classification problems where hand coded rules would be too complex or impossible.
3. Discovering relationships in data (unsupervised).
4. Building models that adapt to constantly changing environments.

3. What is a labeled training set?

A set of data points with some input and some output labels, utilised for supervised learning tasks.

4. What are the two most common supervised learning tasks?

Regression and classification.

5. Name 4 common unsupervised learning tasks

1. Assigning data points to groups (aka clustering).
2. Visualization.
3. Dimension reduction.
4. Association rule learning.

6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?

Reinforcement learning (eg reward-based).

7. What type of algorithm would you use to segment your customers in multiple groups?

Unsupervised clustering algorithm.

8. Would you frame the problem of spam detection as a supervised learning or unsupervised?

Supervised.

9. What in an online learning system?

A model that is constantly updated with new data.

10. What is out-of-core learning?

Allows for training against large datasets that don't fit into memory: learning from a little bit at a time.

11. What type of learning algorithm relies on similarity measure to make predictions?

Instance-basd.

12. Difference between a model parameter and a learning algorithm's hyperparameter?

Hyperparameter is only used during the learning phase - not for predictions.

13. What do model based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?

They search for a line of fit across the data utilise regression. They make predictions by utilising the learned coefficients applied against the data points to predict.

14. 4 main challenges of Machine Learning?

1. Not enough data.
2. Data that doesn't generalise across all cases.
3. Model overfitting.
4. Model unfitting (dupe of not enough data?)

15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name 3 possible solutions?

The model is being overfit on the training dataset. Increase dataset size, simply features and tune hyperparameter using validation set.

16. What is a test set and why should you use it?

A test set is used to evaluate the quality of your model trained on the training data. Usually about 20% of your entire dataset.

17. What is the purpose of a validation set?

To tune hyperparameters.

18. What can go wrong if you tune hyperparameters using the test set?

Overfitting parameters to your test set, I guess?

19. What is cross-validation and why should you prefer it to a validation set?

It's utilised when there is insufficient data to have a validation set.
