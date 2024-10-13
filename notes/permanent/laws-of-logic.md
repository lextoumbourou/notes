---
title: "Laws of Logic"
date: 2023-12-15 00:00
modified: 2023-12-15 00:00
status: draft
---

The **Laws of Logic** are a set of fundamental principles which are the foundation of logical reasoning.

Let $p$, $q$, and $r$ be any three propositions.

| Law                        | Formula                                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Idempotent laws            | $p \land p ≡ p$ <br> $p \lor p ≡ p$                                                                         |
| Identity laws              | $p \land \text{true} ≡ p$ <br> $p \lor \text{false} ≡ p$                                                    |
| Inverse laws               | $p \land (\neg p) ≡ \text{false}$ <br> $p \lor (\neg p) ≡ \text{true}$                                      |
| Domination laws            | $p \lor \text{true} ≡ \text{true}$ <br> $p \land \text{false} ≡ \text{false}$                               |
| Commutative laws           | $p \land q ≡ q \land p$ <br> $p \lor q ≡ q \lor p$                                                          |
| Double negation            | $\neg (\neg p) \equiv p$                                                                                    |
| Associative laws           | $p \land (q \land r) ≡ (p \land q) \land r$ <br> $p \lor (q \lor r) ≡ (p \lor q) \lor r$                    |
| Distributive laws          | $p \land (q \lor r) ≡ (p \land q) \lor (p \land r)$ <br> $p \lor (q \land r) ≡ (p \lor q) \land (p \lor r)$ |
| De Morgan's laws           | $\neg (p \land q) ≡ \neg p \lor \neg q$ <br> $\neg (p \lor q) ≡ \neg p \land \neg q$                        |
| Implication conversion law | $p \rightarrow q ≡ \neg p \lor q$                                                                           |
| Contrapositive law         | $p \rightarrow q ≡ \neg q \rightarrow \neg p$                                                               |
| Reductio ad absurdum law   | $p \rightarrow q ≡ (p \land \neg q) \rightarrow \text{false}$                                               |
