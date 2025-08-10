---
title: A gamut of games
date: 2025-05-31 00:00
modified: 2025-05-31 00:00
status: draft
---

*My notes on article [A Gamut of Games](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/1570) published in 2001 by Jonathan Schaeffer**

## Backgammon

[BKG8.9](../../../../permanent/bkg89.md)

* First effort at building a strong backgammon program was undertaken by Hans Berliner of Carnegie Mellon University.
* In 1979, his program BKG9.8 played an exhibition match against newly crowned world champion Luigi Villa.
* Final score was 7 points to 1 in favor of the computer, with BKG9.8 winning four of five games.
* However, a lot of the success was attributed to random chance, which is part of the nature of backgammon.
* “There was no doubt that BKG9.8 played well, but down the line Villa played better. He made the technically correct plays almost all the time, whereas the program did not make the best play in eight out of 73 non-forced situations.” (Berliner, 1980)

[NEUROGAMMON](NEUROGAMMON.md)

* Neural network approach that won first place in the 1989 Computer Olympiad.

[TD-Gammon](TD-Gammon.md)

* Next program was TD-Gammon, which used a neural network trained via "temporal difference" learning from self-play games.
* Similar to [Deep Q-Network (DQN)](../../permanent/deep-q-networks.md), it takes input as the current board position and outputs an estimated score for the position.
* Contains 160 hidden units, approximately 50,000 weights to be trained.
* Trained on 1.5 million self-play games.
* “Tesauro’s success with temporal-difference learning in his backgammon program is a major milestone in AI research.” (Schaeffer, 2001)

## Checkers

Checkers History

* Both Arthur Samuel and Christopher Strachey worked on a checkers program.
* It beat Robert Nealey in a 1963 exhibition match, and people erroneously concluded that checkers was a "solved" game.
* It was beaten in 1979 by further research at Duke University but still did not reach human-level.

[Chinook](../../permanent/chinook.md)

* In 1989, [Chinook](../../permanent/chinook.md) was developed, initially defeated by the greatest player of all time, Marion Tinsley.
* In 1992, Tinsley beat Chinook 4–2 in a 40-game match.
* In 1994, Tinsley resigned the rematch after 6 drawn games due to health issues; Chinook became world champion.
* Chinook used alpha-beta search with enhancements and large endgame databases (covering all 8-piece positions).
* “CHINOOK is the first program to win a human world championship for any game.” (Schaeffer, 2001)

## Chess

Chess

* Progress in chess was influenced by Ken Thompson, equating [Search Depth](Search%20Depth.md) with chess-program performance.
* Various milestones:
  * **CHESS 4.6** (1978–80) from Northwestern University
  * **BELLE** (1980–82), first U.S. master title
  * **CRAY BLITZ** (1983–84), on a Cray supercomputer
  * **HITECH** and **WAYCOOL** (mid-80s), hardware-based machines
  * **CHIPTEST → DEEP THOUGHT → DEEP BLUE** (1987 onward)
* [Deep Blue](Deep%20Blue.md)
    * Deep Blue beat world champion Garry Kasparov in a 6-game match in 1997.
    * Deep Blue used special-purpose VLSI chess chips to evaluate \~200 million positions per second.
    * “Considering the formidable computing power that DEEP BLUE used... one can only admire the human champions for withstanding the technological onslaught for so long.” (Schaeffer, 2001)
    * “From the scientific point of view, it is to be regretted that DEEP BLUE has been retired... The scientific community has a single data point... the sample size is not statistically significant.” (Schaeffer, 2001)

## Othello

* First major Othello program was Paul Rosenbloom’s **IAGO** (1982), achieving strong early results but only played two games against world-class humans—both losses.
* **BILL** (Kai-Fu Lee and Sanjoy Mahajan, 1990) improved significantly, combining deep search and evaluation-function tuning.
* The best program was **LOGISTELLO** (Michael Buro), which dominated the 1990s.
  * Played and won 6-0 against world champion Takeshi Murakami in 1997, proving computers had surpassed humans.
  * Evaluation function divided the game into 13 phases based on disc count.
  * Used 46 board patterns across all game phases and trained weights using 11 million scored positions.
  * Was table-driven and used deep search plus perfect endgame play.
  * “The gap between the best human players and the best computer programs is believed to be large and effectively unsurmountable.” (Buro, 1997)

## Scrabble

* Early programs emerged in the 1980s.
* **CRAB** won the first Computer Olympiad in 1989, followed by **TYLER** and **TSP**.
* [MAVEN](MAVEN.md)
    * Algorithmic approach to playing Scrabble by Brian Sheppard.
    * By late 1990s, consistently beat top human players.
    * Beat world champion Joel Sherman and runner-up Matt Graham 6–3 in 1998.
    * Beat Adam Logan 9–5 in another 1998 exhibition.
* Used simulations and multiple move generators to reduce a huge search space (700+ legal moves).
* Specialised modules for early game, pre–end-game, and end-game.
* Combined search with rack quality heuristics and deep endgame search.
* Made errors of \~1 point/game vs humans at \~40 points/game. Played nearly perfect Scrabble.
* "The evidence right now is that MAVEN is far stronger than human players.… I have outright claimed... that MAVEN should be moved from the "championship caliber" class to the "abandon hope" class." (Sheppard, 1999)

## Bridge

* Early efforts were weak.
* **GIB** (Matthew Ginsberg) was the breakthrough program in late 1990s.
  * Won the World Computer Bridge Championship and played impressively against human champions.
  * Nearly beat Zia Mahmood and Michael Rosenberg in 1998.
  * Finished 12th in the 1998 Par Contest, scoring 11,210 of 24,000 points.
* Card play is near world-class; bidding is handled via simulations using a database of 7,400 rules.
* Used error-checking and bias to correct for imperfect databases.
* “GIB is well on the way to becoming a world-class bridge player. The program’s card play is already at a world-class level.” (Ginsberg, 1999)

## Go

* Considered the grand challenge of board games.
* Alpha-beta search is ineffective due to large board (19×19) and branching factor.
* Best programs like **GOEMATE** and **GO4++** reach only around 8 kyu (intermediate amateur).
* Programs rely on local pattern-based heuristics and manually coded domain knowledge.
* The **Ing Prize** (\$1.5 million) incentivizes creating a program that beats strong humans.
* Very little high-quality publication or shared code; progress is slow and fragmented.
* “Few of the top programmers have an interest in publishing their methods... The most interesting developments can be learned only by direct communication.” (Müller, 1999)

## Poker

* Strategically complex due to hidden information, bluffing, opponent modeling.
* **ORAC** (1984) had promising exhibition matches but was never rigorously validated.
* **R00LBOT** and **POKI** (University of Alberta) played well on IRC poker servers using simulated opponent modeling.
* **Generic Opponent Modeling (GOM)** used probabilities for hands based on reasonable play.
* **Specific Opponent Modeling (SOM)** adjusted based on observed behaviors via neural nets.
* Even best current programs (as of 2001) are strong intermediate level only.
* “To play poker well, a program needs to... model the opponents... and bluff. Not only does opponent modeling have tremendous value in poker, it can be the distinguishing feature between players at different skill levels.” (Billings et al., 2001)

## Other Games

* Superhuman performance in less common games like **Awari** and **Lines of Action**.
  * Awari nearly solved—databases for all positions with ≤38 stones.
  * **MONA** (Lines of Action) won a world mail-play championship.
* Perfect play (solved) achieved in:
  * **Nine Men’s Morris**, **Connect-4**, **Qubic**, **Go Moku**, **8×8 Domineering**.
  * Used endgame databases and exhaustive search.

## The Future of Computer Games

* Go will continue to challenge researchers for decades.
* New games like **Octi** designed to resist computer strategies (high branching, capability changes).
* AI in modern interactive games (e.g., Quake, Baldur’s Gate, SimCity) is shallow—graphics over intelligence.
* Potential for game AI to evolve toward human-level interaction, driven by the commercial gaming industry.
* "Computer games are the ideal application for developing human-level AI." (Laird & van Lent, 2000)
* "Games are ideal domains for exploring the capabilities of computational intelligence. The rules are fixed, the scope of the problem is constrained, and the interactions of the players are well defined." (Schaeffer, 2001)
