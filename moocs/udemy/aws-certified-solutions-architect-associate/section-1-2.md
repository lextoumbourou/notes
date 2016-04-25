# AWS Certified Solutions Architect Associate

## Course requirements

* An AWS Free Tier Account.
* A computer with an SSH terminal.
* A domain name (optional).

## Exam Blue Print

* Exam for "AWS Certified Solutions Architect Associate".
* Objectives:
  * Designing highly available, cost efficient, fault tolerant, scaleable systems. (60%)
  * Implementing/Deploying. (10%)
  * Data security. (20%)
  * Troubleshooting. (10%)
* About the exam:
  * 80 minutes in length.
  * 60 questions (can change).
  * Multiple choice.
  * Pass mark based on "bell curve" (it moves around).
  * Aim for 70% to pass.
  * Qualification only valid for 2 years.
  * Scenario based questions.

***

## History of AWS so far

* 2003 - Chris Pinkman & Benjamin Black present a paper on what Amazon's own internal infrastructure should look like.
* Suggested selling it as service and prepared a business case.
* SQS officially launched in 2004.
* AWS officially launched in 2006.
* 2007 over 180,000 devs on the platform.
* 2010 all of amazon.com moved over.
* 2012 First Re-Invent Conference.
* 2013 Certifications Launched.
* 2014 Make commitment to achieve 100% renewable energy usage.
* 2015 make revenue public: $6 billion USD per annum, growing 90% year on year.

****

## Concepts & Components (Part 1)

* AWS Global Infrastructure
  * December 2015:
    * 11 Regions.
    * 30 Availablility Zones.
  * 2016:
    * 5 More Regions.
    * 10 More Availability Zones (2016)

* What is a region? What's an availablility zone?
  * A region is a geographic area.
  * Region consists of 2 (or more) availability zones.

* An Availability Zone:
  * A data center

* Edge Locations
  * CDN End Points for CloudFront.
  * There are more Edge Locations than Regions.

* Regions overview:
  * North America:
    * US East (Northern Virginia) - 5 AZ
    * US West (Oregon) - 3 AZ
    * US West (Northern California) - 3 AZ
    * AWS GovCloud (US) Region. - 2 AZ
  * South America
    * Sao Paulo Region - 3 AZ
    * Edge Location: Rio de Janeiro, Sao Paulo
  * Europe / Middle East / Africa
    * EU (Ireland) - 3 AZ
    * EU (Frankfurt) - 2 AZ
  * Asia Pacific
    * Asia Pacific (Singapore) - 2 AZ
    * Asia Pacific (Sydney) - 2 AZ
    * Asia Pacific (Tokyo) - 3 AZ
    * China (Beijing) - 2 AZ

## AWS Platform

* To pass exam, you should know names of service and what they do. Don't need to know in the outs of everything except the core services.

### Networking

* VPC
  * basically a "virtual data center"
* Direct Connect  - allows to connect without an internet connection
* Route 53 - AWS DNS service.
  * Name is hybrid of "route 66" and DNS' TCP port 53.

### Compute

* EC2
  * virtualisation in the Cloud.
* EC2 Container Service
  * Fast container manager service.
* Elastic Beanstalk
  * Sort of like AWS for beginners. Upload your code and AWS will figure out how to run it.
* Lambda
  * Server-less architecture.

### Storage

* S3
  * Object-based storage (as opposed to block-based).
  * Big part of exam
* Cloud Front
  * Different edge locations throughout the world.
  * Files can be cached locally for users.
* Glacier
  * Archiving server in the cloud.
  * Slow to access.
  * Cheap.
* EFS
  * NAS in the Cloud.
  * Block-based storage.
* Snowball
  * Send your HD into AWS Cloud and they'll load it for you.
* Storage Gateway
  * Can replicate data from your own office into AWS.

### Database

* RDS
  * "Relation database services".
  * Bunch of choices for relational databases including Amazon's Aurora.

* DynamoDB
  * AWS's non-relational DB.

* Elasticache
  * AWS's cache product.

* Redshift
  * Amazon's data warehousing service.

* DMS
  * Database migration services.
  * Just announced, currently in beta.

### Analytics

* EMR
  * Elastic map reduce.

* Data Pipeline
  * Way of moving data from one service to another.

* Elastic Search
  * Hosted ES (not in exam). 

* Kinesis
  * Streaming data in AWS.

* Machine Learning

* Quick Sight
  * Competitor to "Cognos"

### Security and Identity

* IAM
  * Identity Access Management.
  * Big part of exam.

* Directory Service

* Inspector
  * Not in exam.
  * Give suggestions on how to secure environment.

* WAF
  * Not in exam.

* Cloud HSM (Hardware Security Module)

* KMS
  * Key management service

### Management Tools

* Cloud Formation
* Cloud Watch
  * Metrics for AWS services.
* Cloud Trail
  * Way to audit AWS.
* Opsworks
  * Configuration management tool using Chef.
* Config
  * Security and governance tool.
* Service Catalog.
* Trusted Advisor.
  * Tells you on ways you can save money and increase security.

### Application Services

* API Gateway
  * Bunch of tools for managing an API.
* AppStream
* CloudSearch
  * AWS's search product.
* Elastic Transcoder
  * For transcoding media files.
* SES
  * Send transactional emails and marketing messages.
* SQS
  * First service launched by AWS.
  * Important for exam.
* SWF
  * "Simple workflow service"

### Developer tools

* CodeCommit
  * AWS's version of Github.
* CodeDeploy
  * Automate code deployments to any instance.
* CodePipeline
  * Continuous delivery service.

### Mobile Services

* Mobile Hub
  * Build and test mobile apps
* Cognito
  * Save mobile user data in AWS Cloud.
* Device Farm
  * Enables you to improve quality of web apps
* Mobile Analytics
  * Track new vs returning users etc.
* SNS
  * Simple notification service.

### Enterpise Applications

* WorkSpaces
* WorkDocs
  * For sharing docs.
  * Dropbox for Enterprise.
* WorkMail
  * Webmail.

### Internet of Things

* Internet Of Things
  * Service just announced.

## Mind Overload? Don't Stress!

* Don't need to know all services inside out.
