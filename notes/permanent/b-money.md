---
title: B-Money
date: 2025-03-10 00:00
modified: 2026-09-26 08:57
status: draft
---

**b-money** was a proposal created by Wei Dai about an "anonymous, distributed electronic cash system", which was referenced in the [Bitcoin: A Peer-to-Peer Electronic Cash System](bitcoin-a-peer-to-peer-electronic-cash-system.md) paper.

I personally don't subscribe to the idea that "government is not temporarily destroyed but permanently forbidden and permanently unnecessary". Nonetheless, the technology laid the foundation for an interesting and mysterious software, [Bitcoin](bitcoin.md), which has very much changed the course of history forever.

Dai described two key protocols:

1. Idealistic Protocol: Everyone maintains a copy of all financial records, with money created through computational work. Transfers occur through broadcast messages, with contracts enforced through collective verification and arbitration.
2. The Practical Protocol: A subset of participants ("servers") maintain the ledgers, publish them regularly, and are kept honest through security deposits and public verification.

## Idealistic Protocol

Every participant maintains a separate database of how much money belongs to each pseudonym.

### 1. Creation of money

People can "create money" by broadcasting a solution to a previously unsolved computational problem. It must be easy to determine how much computing effort it took to solve the problem, and the solution must otherwise have no value, either practical or intellectual.

The number of monetary units created is equal to the cost of the computing effort in terms of a standard basket of commodities. For example if a problem takes 100 hours to solve on the computer that solves it most economically, and it takes 3 standard baskets to purchase 100 hours of computing time on that computer on the open market, then upon the broadcast of the solution to that problem everyone credits the broadcaster's account by 3 units.

### 2. Transfer of money

If Alice (owner of pseudonym K_A) wishes to transfer X units of money to Bob (owner of pseudonym K_B), she broadcasts the message "I give X units of money to K_B" signed by K_A. Upon the broadcast of this message, everyone debits K_A's account by X units and credits K_B's account by X units, unless this would create a negative balance in K_A's account in which case the message is ignored.


### 3. The effecting of contracts

A valid contract must include a maximum reparation in case of default for each participant party to it. It should also include a party who will perform arbitration should there be a dispute. All parties to a contract including the arbitrator must broadcast their signatures of it before it becomes effective. Upon the broadcast of the contract and all signatures, every participant debits the account of each party by the amount of his maximum reparation and credits a special account identified by a secure hash of the contract by the sum the maximum reparations. The contract becomes effective if the debits succeed for every party without producing a negative balance, otherwise the contract is ignored and the accounts are rolled back. A sample contract might look like this:

* K_A agrees to send K_B the solution to problem P before 0:0:0 1/1/2000.
* K_B agrees to pay K_A 100 MU (monetary units) before 0:0:0 1/1/2000.
* K_C agrees to perform arbitration in case of dispute.
* K_A agrees to pay a maximum of 1000 MU in case of default.
* K_B agrees to pay a maximum of 200 MU in case of default.
* K_C agrees to pay a maximum of 500 MU in case of default.

### 4. The conclusion of contracts

If a contract concludes without dispute, each party broadcasts a signed message "The contract with SHA-1 hash H concludes without reparations." or possibly "The contract with SHA-1 hash H concludes with the following reparations: ..." Upon the broadcast of all signatures, every participant credits the account of each party by the amount of his maximum reparation, removes the contract account, then credits or debits the account of each party according to the reparation schedule if there is one.

### 5. The enforcement of contracts

If the parties to a contract cannot agree on an appropriate conclusion even with the help of the arbitrator, each party broadcasts a suggested reparation/fine schedule and any arguments or evidence in his favor. Each participant makes a determination as to the actual reparations and/or fines, and modifies his accounts accordingly.





