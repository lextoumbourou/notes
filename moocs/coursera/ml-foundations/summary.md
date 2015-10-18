# Summary

## Deploying ML as a service

* Deployment system example

  * Train model.
  * Deploy models and make predictions.

* Lifecycle of ML in Production

  * Consider A / B testing for different models.
  * Monitoring models with real users.

* Evaluating model quality

  * Metrics to evaluate models offline: test data. Sum of squared errors etc.
  * Online: user feedback, metrics etc. 

* Update ML models

  * Why update?

    * Trends and users change over time.
    * Model performance drops.

  * When to update?

    * Track statistics over time.
    * Monitor both offline and online metrics.
    * Update when offline metric diverges from online metrics or not achieving targets.

* A / B Testing: Choosing between ML models

  * Look at click-through rates between different models.

* Other production considerations

  * Versioning
  * Provenance
  * Dashboards
  * Reports
