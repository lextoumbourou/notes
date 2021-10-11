---
title: Modern information theory
date: 2021-07-18
status: draft
---

## Ancient information theory

### What is information theory?

* About "fundamental particle" of all forms of communication.
* Language allows you to take a thought or mental object and break down into "conceptual chunks"
* Chunks are externalised using signals or symbols
    * Humans use sound and physical action
    * Machines use streams of electrical vibrations
    * All forms of one thing: information
* Information allows one mind to influence another
* Information can be measured or compared using a measurement called [[Information Entropy]]
* We can describe exactly how much information a source has using a unit called: the bit - a measure of surprise
* A bit linked to simple idea: answer to yes or no question
    * Think as the language of coins
* How is information measured?
    * Information theory holds the answer

### Origins of written language

* 50k years ago there was a explosion of diverse cultural artifacts, including instruments and new tools
    * Humans developed ability to externalise inner thoughts
* Alice travels back in time to find Bob
    * Bob leaves behind a handprint
    * A handprint contains very little information, just that he was here and could return
    * Alice knows Bob is intelligent, but cannot read or write in any language
    * Alice paints picture using natural materials
    * Renders animal she is tracking, as clue where she's travelling
    * Ancestors used cave paintings to make pictorial renderings of their surrounding
    * When Bob returns, he finds paintining and goes to river where he thinks she may be
    * He doesn't find her. Paints a picture showing where he's going next.
        * Message contains 3 distinct objects: middle, river, west
        * Uses simplified pictures to represent them
        * [[Pictogram]] (03:03)
            * A symbol that represents an object in natural form
            * Pictograms step in development of writing
        * Hard to draw pictures of abstract concepts: calm, ferocious, middle
        * [[Ideogram]] (03:59)
            * Conceptual picture of abstract idea
        * Bob uses picture of setting sun for West
        * Bob combines individual symbols to create new message
            * Meaning + meaning = new meaning
        * Early artifacts of merging found in Mesopotamia
        * Over time, Bob learns Alices language allow them to use same oral language to communicate concepts and ideas
        * Root of more powerful written languge starts with writing name:
            * Disassocaites the sound from picture for name Alice
            * Combines math symbol for All and picture of ice: all + ice
                * Name has nothing to do with symbols: sound + sound = new meaning
        * [[Rebus Principal]] (06:16)
            * Sound + sound = new meaning
            * Example from earliest hyrohliphics found: Narmer
                * Inscription of name as fish and chisel: Nar, mer
                    * 2 sounds separated from picture giving meaning


### History of the alphabet

* Informally think of information as some message, stored or transmitted, using some medium
* When humans begin building writing systems, we divided world in finite living symbols.
    * Any written language can be thought of in this way.
* Messages are formed by arranging symbols in specific patterns.
* 2 ancient writing systems in 3000 BC
    * In ancient Eygpt, Hieroglyphics:
        * Reserved for governmental, magical and religious purposes.
        * We practiced by a few experts called scribes: not used by common people
        * Symbols in 2 categories:
            * Word signs: symbols that rep a single meaningful concept
            * Sound signs: chunks of sound
        * Total number of symbols was over 15k.
        * Much smaller sound signs: 140
            * Only 33 represent constontants
        * Medium used was primarily rock: messages can travel into the future
        * Another technology emerging technology from crops called papyrus 
            * Ideal for sending messages across greater spaces
            * This introduced more people to writing
            * Symbols evolved to allow faster "cursive writing"
                * New script called [[Hieratic]]
                * Number of common symbols shrink to 700
                * By escaping from heavy medium of stone, though gained "lightness"
            * More writing by hand, accompanied by "secularisation" of writing, thought and activity.
            * Lead to new writing system called [[Demotic Script]] around 650 BC
                * Shift to phonetic symbols over word symbols
                    * Meant children could be taught to write at a young age.
        * Same pattern in other cultures
            * In 3000 BC in Mesopotamia had writing system called [[Cuneiform]] used for finance purposes (tracking debt and surplus commodities before coins)
            * Writing system used by Summarians, which had 2000 symbols in use that could also be divided into word signs and sound signs
            * (05:35) [[Akkadian]] gradually replaced Sumerian as spoken language
                * Earliest known dictionary from 2300 BC
                    * Contains word list in Sumerian and Akkadian (discovered in modern Syria)
                        * Reduced symbols to 600 by moving to sound signs
            * As writing systems escaped formal and spread to the people
                * Invention of new writing systems for the people
            * Discovery in 1700 BC of the Sinai inscriptions
                * Each picture denotes constinant sounds: no word signs used
                    *  Revolution: creating meaning with sound signs only
            *  By 1000 BC, we have Phoenician Alphabet, which emerges along Mediterrainian
                *  Based on principal that one sign = one constantent.
                *  Used to write northern semetic language containing 22 symbols total
                *  Secret power: did not need semetic speech to work
                    *  Letters could be fitted to diverse tounge across world
                *  Was source of greek and roman alphabet we know today
            *  Alphabet is a powerful method for transmitting and storing information
            *  Information is just a selection from a collection of possible symbols
            *  Over time: looked for faster ways to transport information
                *  When we try faster than humans or animals can travel, engineering problem present themselves.

### The Rosetta Stone

* Currently in the British Musem, London.
* Found in 1799.
* Stone had same statement said 3 times in different languages:
    * Hieroglyphics
    * Demotic - common language used by Eygpytian people
    * Ancient Greek (language of government in 200 BC - Alexanda the Great had conquered Egypt and setup Greek rule)
* Allowed us to read and understand and hieroglyphics.
* Taught us that hieroglyphic symbols weren't pictorial, but phonetic: all the things that look like pictures, actually represent sounds.
* How?
    * Saw Cartouches which held names of rulers. By recognising ruler's name name Ptolemy V in 3 languages, could unlock hieroglyphics.
* Found by Napolean
    * Found by archeaologist in Eygpt
* British defeated Napolean and brought the stone to Britain.

### Source encoding

* If you attach tin can to each end of wire, can send voices along wire.
    * Key problem: noise
* When high wind, can't hear signal over [[Noise]].
* Solution?
    * Pluck the wire and use vibrations - better over noise
    * How do you encode message as plucks?
* Play board game analogy
    * Tackle common messages first: outcome of 2 dice roles
    * Messages sending a selection from a finite number of symbols: 11 possible numbers
        * Called: [[Discrete Source]]
    * Simplest method: send results as number of plucks
        * To send 3, 3 plucks etc
    * Soon realised that it takes much longer than it needs to.
    * 2 plucks per second is the max amount they can send: [[Capacity]] for sending information
    * Most common role is 7. Takes 3.5 seconds to send on average.
    * Odds of each number follows a simple pattern.
    * Alice realised odds of each number sent follow a pattern (normal distribution of values)
        * 1 to role 2, 2 ways to role 3... 6 ways to role 7 ...
        * Make 7 one pluck, 6, 2 plucks ... 
    * Simple change, allows to send more [[Information]] on same time on average.
    * Impossible to send shorter message with identical plucks.
* New idea:
    * vary speed of plucks.

### Visual telegraphs (case study)

* Signal fire one of oldest technology for transmitting information.
* Dating back to controller use of fire.
* Allows one person to influence another's [[Belief State]] over a distance.
    * Switch between one or more belief states
    * Of great important to military powers
* Greek myth of Cadmus, a Phoenician prince who introduced the phonetic letters to Greece.
* The Greek alphabet borrowed from Phonetian letters as well as light and cheap ppyrus
    * Allowed for transfer of writing from priestly to military glass
    * Greek military history has evidence of messages through signal torches
* Poylbuia a Greek historian born 200BC, wrote "The Histories".
    * About communication tools of the tool
    * Fire signals limitation was clear to him
        * Fire signal works when space of message signals is small: enemy has arrived or not.
        * When [[Message Space]], total number of possible messages, grows - need to communicate many messages
    * Technology by Aeneas Taciticus, earliest Greek writers, from fourth century
        * People about to communicate messages by fire signal get a vessels of equal depth
        * Put a rod in the liquid
        * When torch is raised by both thyey start
        * Remove torhc, then wait until event is reached to stop flow
        * Equal waters levels denotes singal message
        * Uses difference in time to communicate messages.
        * Limited by speed
    * Democritus had another method
        * Polybius Square
            * Two people separated by distance have 10 torch in 2 groups of 5.
            * Sender lights some torches from each group
            * Number defines row position in alphabetic grid and 2nd column
            * Allows for 5*5 messages
    * 6th century BC indian medical text described combinatorial situation:
        * Give 5 spices, how many different tastes can you make?
        * Can be broken into 6 questions, creating a tree.
        * Given n yes or no questsions, there are 2^n possible answer sequences.
* 1605 - Francis Bacon explained how this idea can let one send all letters of alphabet using single difference
    * The transposition of 2 letters by five placings will be sufficient for 32 differences (2**5)
* Idea of using single difference to communicate the alphabet, became popular in 17th century due to telescope
    * Allowed magnification power to jump significantly
* Robert Hooke in 1684
    * With a little practice the same character can be seen at Paris within a minute
* Lord George Murray's shutter telegraph was Britain's reaction to bonaparte's threat to England
    * Composed of 6 shutters, which could be opened or closed .
        * That gives us enough for all letters and digits
        * Each observation can be 1 of 64 paths through decision tree.
* An observation in 1820, led to new technology that allowed much bigger communication distance, launching the information age.

## Electrostatic telegraphs (case study)

* People tried to find methods for sending sparks over long distances, since Benjamin Franklin's early experiments.

## The battery and electromagnetism





Notes from [Modern Information Theory](https://www.khanacademy.org/computing/computer-science/informationtheory/moderninfotheory) on Khan Academy.

## Symbol rate

* How should we measure information for any communication system? Human, animal, alien.
* In late 19th century focused on speed.
    * One goal: input letters and have machine automate "secondary source"
* Baudot Multiplex System (1874)
    * 5 keys which could be played in any combination
    * 2^5
    * Operator plays letters, machine outputs pulse stream representing the letters
    * Sequence was divided using a clock
    * Limiting speed not the clock: physically limited by minimum spaces between impulses (pulse rate)
* Sending pulses too fast = intersymbol interfereces
* Fundemantl limit of how far we cn squeeze 2 pulses together
* Symbol rate = how far aparts pulses need to be to ensure we don't have intersymbol interference
* Signal event - change from one state to another

## Introduction to channel capacity

* Another way to increase capacity: increase number of different signaling events.
* Difference between signalling events needs to be great enough that noise doesn't cause you to bump to a different signal
* Capacity:
    * Symbol rate (current referred to as Baud rate) - n
    * Difference - s
* Parameters can be thought of as decision tree of possibilities
    * The number of leaves is size of the message space
    * Width of the base of the trees
    * s^n = message space
* Example: 
    * 2 letters on note = 26 letters (difference), 2 letter per note (symbol rate)
        * 26^2 = 676 messages

## Measuring information

* What's the price of message?
    * Price is how long it takes to transmit it?
    * How do you measure types of information using a common metric?
* Imagine you can only one bit (binary digits) of information which are answers to questions
    * Challenge: send data to answer question
    * For one flip coin: one question (is it heads?)
    * For 10? 10 questions
    * Letters:
        * we could ask questions that elimate half. Is it less than N?
        * is it less than G?
        * is it less than C
        * is it greated than D
        * Is it E?
        * Takes 5 questions
        * 2^(# questions) = message space
* How do you calculate expected number of questions?
    * 2^? = 26
    * log_2(26) = 4.7
* Unit is called the bit
* In paper by Ralph Hartley he defined information using symbol H:
    * $H = n \log(s)$

## Origin of Markov chains

To do.

## A mathematical theory of communication


## Information entropy

* If you had 2 machines that output an alphabet, with 2 different probability distributions, which one produces more information?
    * Machine 1 generates randomly:
        * P(A) = 0.25
        * P(B) = 0.25
        * P(C) = 0.25
        * P(D) = 0.25
    * Machine 2 generates accroding to probability: 
        * P(A) = 0.5
        * P(B) = 0.125
        * P(C) = 0.125
        * P(D) = 0.25
     * Claude Shannon rephrased the question: what is the mininum yes or no questions you'd expect to ask to predict the next symbol?
     * You might start by asking a question that divides the space in half (is it A or B?).
         * Now you can eliminate half the possibilities. 
     * Uncertainy of Machine 1 is 2 questions.
     * Uncertainy of Machine 2 is 1.75 questions on average. Since by asking is it A, 50% of the time it's only 1 question.
     * We say that Machine 2 is producing less information, because there's less "uncertainy" or surprise*
    * Claude Shannon calls this measure of average uncertainy: [[Information Entropy]] and uses letter `H` to represent.
    * Unit of entropy is based on uncertainty of fair coin flip.
        * Called the bit
    * Equation is $H=\sum\limits_{i=1}^{n} p_i \log_2 (1/p)$ which can be rewritten as $H=-\sum\limits_{i=1}^{n} p_i \log_2 (p)$
    * Entropy is max when all outcomes are equally likely
    * When predictability is introduced, entropy decreases.