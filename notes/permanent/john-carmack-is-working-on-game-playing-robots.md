---
title: John Carmack is working on game-playing robots
date: 2025-05-23 00:00
modified: 2025-05-23 00:00
summary: "on John Carmack's Upperbound 25 Talk Notes"
cover: /_media/atari-realtime.png
category: reference/talks
tags:
- GamePlayingAI
- LimitationsofLLMs
---

John Carmack shared his notes from a recent talk at an AI conference about the research he and his team of six are conducting at his new AGI company, Keen Technologies.

I'm a big fan of Carmack. I grew up on his games, from Commander Keen to Quake. I love his approach to engineering, especially after reading Masters of Doom about the early years of id Software.

Keen Technologies is revisiting a seemingly largely solved problem: playing Atari Games.

In 2013, Deepmind described the first deep-RL approach to playing Atari games that achieved human-level performance just from reading pixels (on some games, at least). Then, in 2020, Deepmind returned and smashed their own record by beating nearly all Atari games to human-level performance in their Agent57 approach. See [Playing Atari with Deep Reinforcement Learning](../reference/papers/playing-atari-with-deep-reinforcement-learning.md).

So you might think Atari is completely solved, so what's left to research? Well, has it been solved with **robots**??

> "A reality check for people that think full embodied AGI is right around the corner is to ask your dancing humanoid robot to pick up a joystick and learn how to play an obscure video game."

Carmack's team has built a system that uses servo motors to control a physical controller, which can learn to play Atari games in real-time, using just camera input as a signal.

I think the intersection of AI research and hardware will be a boon for the capability of embodied AI in general. Despite the rapid progress of LLM technology deployed to consumers, it's yet to meaningfully demonstrate improvements in embodied agents, even though vision understanding capabilities are getting so good.

The real-time aspect is really important: "Reality is not a turn-based game", he says. So far, all the prior work with game playing, and even with the LLM interactions we've seen, have assumed a turn-based approach. You write a prompt, the LLM writes a response, and so on; even with the LLMs that operate using voice, the experience is largely the same. Say some things, and wait for a response, then say something else, which is a different style of conversation to the one we'd have with a person, with pauses, and interruptions, etc. Some of the boundaries to human-like conversations are clearly in the hardware.

Secondly, he's choosing not to focus on pre-trained LLMs, putting him at odds with nearly everyone else in AI right now.

> "I believe in the importance of learning from a stream of interactive experience, as humans and animals do."

I think there's likely a lot of value in continuing to explore techniques outside of LLMs, as there's a distinct possibility that the progress of AI will be stuck in a local minima for a while, with so much investment and effort thrown into LLMs, just because of much juice is clearly left to extract from them. It's good to have someone so talented and driven thinking about other possibilities. Still, I don't see why LLMs couldn't be trained on a big enough corpus of robot game-playing that they could demonstrate the same capability that Carmack is going for. This point he does acknowledge, with a references to Sutton's Bitter Lesson. Incidentally, Sutton joined the team at Keen technologies in 2023.

Anyway, who am I to question Carmack? Let him cook.

---

There's a lot of other really interesting details in there. I highly recommend reading the notes.

* [Talk notes](https://docs.google.com/document/d/1-Fqc6R6FdngRlxe9gi49PRvU97R83O7ZTN_KFXo_jf0/edit?tab=t.0#heading=h.628l6khl68xe)
* [Slides](https://docs.google.com/presentation/d/1GmGe9ref1nxEX_ekDuJXhildpWGhLEYBMeXCclVECek/edit?slide=id.p#slide=id.p)
* [Tweet](https://x.com/ID_AA_Carmack/status/1925710474366034326)