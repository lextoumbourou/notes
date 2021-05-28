---
author: John Ferguson Smart
aliases: ["BDD in Action"]
---

#  BDD in Action

## 1. Building software that makes a difference

* 42% of software projects were delivered late, ran over budget or didn't deliver all requests features, according to 2011 edition of Standish's Group's annual CHAOS report. 21% of all projects were cancelled entirely.
* TDD does reduce defects and increase software quality
    * Unit tests in TDD can become coupled to spefic implementations and difficult to change and manage.
* Behavior-Driven Development (BDD)  draws on TDD and Domain-Driven Design (DDD)
    * Provides common language based on simple, structured sentences expressed in native language to facilite comms between stakeholders and project team members.
* Dan North [created BDD](http://dannorth.net/introducing-bdd/).
    * He observed that making unit tests names full sentences prefixed by "should" helped developers write meaningful tests.
* Domain-Driven Design, introduced by Eric Evans, promotes a language that business analysts can use to define requirements unambiguosly:
       
       - Given a customer has a current account
       - When the customer transfers funds from this account to an overseas account
      - Then the funds should be deposited in the overseas account
       - And the transaction fee should be deducted from the current account
       
* This notation has evolved into something now commonly called "Gherkin"
    * Dan North wrote the first dedicated test automation library called "JBehave" that automated this acceptance criteria.
* Other names for BDD:
    * Acceptance-Test-Driven Development
    * Acceptance Test-Driven Planning
    * Specification by Example
    * Note: other examples are slight variations but aim to achieve the same goal.