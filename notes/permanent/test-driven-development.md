---
title: Test-Driven Development
date: 2023-11-13 00:00
modified: 2023-11-13 00:00
status: draft
---

Test-Driven Development (TDD) is an approach to software development where we write tests before "production" code. This approach ensures we continually build a test suite and define our desired functionality as we write our code. The process typically follows a cycle known as "Red-Green-Refactor":

- First, write a test that fails (Red).
- Write code to make the test pass (Green).
- Finally, refactor the code, ensuring the tests continue to pass.

TDD for me is a useful tool, some problems are much much easier to write tests first. Particularly when you have examples of the inputs and outputs readily available.

## [[The Three Laws of TDD]]

If you want to get particularly dogmatic about it, [3 laws of TDD are according to Robert Martin](http://butunclebob.com/ArticleS.UncleBob.TheThreeRulesOfTdd) are:

* You may not write production code until you've written a failing unit test.
* You may not write more of a unit test than is sufficient to fail.
* You may not write more production code than is sufficient to pass the test.

However, lots of code should not be written in this way. Most of the time the inputs and outputs are not clear, and we write the code to move us toward the next solution so we can learn more about the problem.