A well-designed API (application programming interface), is one in which [[Observabled Behaviours]] are entirely in the [[Public API]], and implementation details are part of the [[Private API]].

According to Khorikov, the simplest way to tell if a class leaks a implementation detail into observable behaviour is if number of operations client needs to perform to achieve a goal is > 1.

Good [[Encapsulation]] protects against [[Invariant]] volations, especially as the complexity of the code base increases.

*"You cannot trust yourself to do the right thing all the time - so, eliminate the very possibility of doing the wrong thing"*

Related to Martin Fowler's [[Tell Don't Ask]] pattern.

---

Tags: #SoftwareEngineering
Reference: [[Unit Testing Principles, Practices and Patterns#5 Mocks and test fragility]]