---
title: B-Money
date: 2025-03-10 00:00
modified: 2026-09-27 06:43
status: draft
summary: Wei Dai's 1998 proposal for anonymous electronic cash, with computational money creation and two approaches to maintaining account balances.
tags:
- Bitcoin
- Cryptography
---

**b-money** was Wei Dai's 1998 proposal for an anonymous, distributed electronic cash system [@daiBMoney1998]. It was referenced in the [Bitcoin](bitcoin.md) whitepaper [@nakamotoBitcoinPeertoPeerElectronic2008].

I personally don't subscribe to Dai's idea that "government is not temporarily destroyed but permanently forbidden and permanently unnecessary" [@daiBMoney1998]. Nonetheless, the proposal helped lay the foundation for an interesting and mysterious piece of software, [Bitcoin](bitcoin.md), which has very much changed the course of history forever.

Dai described two protocols [@daiBMoney1998]:

1. Everyone maintains account balances, with money created through computational work. Transfers occur through broadcast messages, with contracts handled through deposits and arbitration.
2. A subset of participants ("servers") maintain the ledgers and publish them regularly, with security deposits and public verification intended to keep them honest.

## First protocol: everyone maintains balances

Every participant maintains a separate database of how much money belongs to each pseudonym. Dai calls this protocol impractical because it assumes a synchronous, unjammable anonymous broadcast channel. Both protocols assume an untraceable network and public-key pseudonyms [@daiBMoney1998].

### 1. Creation of money

People can "create money" by broadcasting a solution to a previously unsolved computational problem. It must be easy to determine how much computing effort it took to solve the problem, and the solution must otherwise have no value, either practical or intellectual.

The number of monetary units created is equal to the cost of the computing effort in terms of a standard basket of commodities. For example, suppose a problem takes 100 hours to solve on the computer that solves it most economically. If 100 hours of computing time costs 3 standard baskets on the open market, everyone credits the broadcaster's account with 3 units when they broadcast the solution [@daiBMoney1998].

### 2. Transfer of money

If Alice (owner of pseudonym `K_A`) wishes to transfer `X` units of money to Bob (owner of pseudonym `K_B`), she broadcasts a message giving `X` units to `K_B`, signed with her private key. Everyone debits Alice's account by `X` units and credits Bob's account by `X` units. If this would create a negative balance in Alice's account, the message is ignored [@daiBMoney1998].

### 3. Creating contracts

A valid contract must include a maximum reparation in case of default for each party. It should also include a party who will perform arbitration should there be a dispute. All parties, including the arbitrator, must broadcast their signatures before the contract becomes effective.

Upon the broadcast of the contract and all signatures, every participant debits each party's account by their maximum reparation. The sum goes into a special account identified by a secure hash of the contract. The contract becomes effective if the debits succeed for every party without producing a negative balance. Otherwise, the contract is ignored and the accounts are rolled back [@daiBMoney1998].

Dai's sample contract specifies:

* `K_A` agrees to send `K_B` the solution to problem `P` before 1 January 2000.
* `K_B` agrees to pay `K_A` 100 MU (monetary units) before the same deadline.
* `K_C` agrees to perform arbitration in case of dispute.
* The maximum reparations in case of default are 1,000 MU for `K_A`, 200 MU for `K_B` and 500 MU for `K_C` [@daiBMoney1998].

### 4. Concluding contracts

If a contract concludes without dispute, each party broadcasts a signed message identifying the contract by its hash and stating the agreed reparations, if any. Once all signatures are broadcast, every participant returns each party's deposit, removes the contract account, then credits or debits each party's account according to the agreed reparation schedule [@daiBMoney1998].

### 5. Enforcing contracts

If the parties cannot agree on a conclusion, even with the arbitrator's help, each party broadcasts a suggested reparation or fine schedule and any supporting arguments or evidence. Each participant decides the actual reparations or fines and modifies their own account database accordingly [@daiBMoney1998].

## Second protocol: servers maintain balances

Only a subset of participants, called servers, maintain the account databases. People involved in a transaction check that a randomly selected subset of servers has processed it. Servers post security deposits that can fund penalties or rewards for proof of misconduct. They periodically publish and commit to their money-creation and ownership databases. Participants check their balances and verify that total balances do not exceed the money created. Dai describes this as more practical, but it still requires some trust in the servers [@daiBMoney1998].

## Alternative money creation

Dai also identifies a problem with pricing computational work: hardware improvements can make cost estimates inaccurate or outdated. His appendix proposes negotiating money-creation quotas for each period and auctioning the new units to bidders who submit solutions with the highest nominal computational cost per unit [@daiBMoney1998].
