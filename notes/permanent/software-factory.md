---
title: "Software Factory"
date: 2026-02-08 00:00
modified: 2026-02-08 00:00
cover: /_media/software-factory-cover.jpeg
summary: "is the future of engineering verification?"
tags:
- AIAssistedSoftwareEngineering
- SoftwareEngineering
---

**Software Factory** [^1] refers to the idea of completely abandoning the notion of writing code, even reviewing it, leaving engineers to manage the goal and validate the correctness of the system. Effectively, developers become system-level QA engineers, writing specs and unblocking agents.

StrongDM defines a set of principles for the Software Factory approach [^2], which requires thinking in terms of "**The Loop**": the model starts with a seed, iterates, validates, receives feedback, and continues until all the "holdout scenarios" pass. With that paradigm, the question for the engineer is how to structure the problem so that each criterion can be validated (without return-true-in-tests cheating) and can receive meaningful feedback to guide its further development. StrongDM goes as far as building "Digital Twin Universes": entire replicas of all the tools like Jira, Okta, Google Sheets, etc., that their software integrates with, which they use to test their scenarios exhaustively.

For a software engineer who's made a living writing software by hand for many years, it's very anxiety-inducing to hear people talking about developing software like this. Clearly, fewer people will be involved in achieving the same outcomes (although our ambition for outcomes will likely increase significantly). But I can't help but see some inevitability in this paradigm in future. Because, even before I knew it was a thing, I've been seeing it as a natural way to solve problems in the Opus 4.5+ era of agentic software development.

For example, I wanted an MLX (basically, PyTorch for Apple Silicon hardware) version of Demucs, one of the best audio stem-splitting models available. Without knowing a term for it, more just because I was lazy, I set up a software factory: I gave the model context of the reference implementation (the original PyTorch implementation), and the validation scenario: given a range of audio inputs I selected, the MLX outputs must match the PyTorch outputs within a numerical tolerance. I gave the agent an `IMPLEMENTATION_NOTES.md` file (see [Spec-First LLM Development](spec-first-llm-development.md)) to serve as both the plan and working memory, to be updated as it goes. I reviewed the initial plan to check we agreed on the success criteria, then left it to work. And it did work. The results are [here](https://github.com/lextoumbourou/mlx-demucs).

In another project, I wanted to integrate [Pocketsmith](https://pocketsmith.com) (my budgeting tool) into [OpenClaw](openclaw.md) as a "skill". I pointed the agent at several existing OpenClaw skills as concrete exemplars (which StrongDM calls "Gene Transfusion"), gave it the Pocketsmith API documentation, provided a basic indication of how I wanted to use it, told it how to plan and track its tasks, and let it work. The verification was mainly just me testing it and making sure I was satisfied with the interface it had constructed. This skill is just for my own personal use, so I'm not so worried about exhaustive verification - the surface for things to go disastrously wrong is limited. The results are [here](https://github.com/lextoumbourou/pocketsmith-skill). I also integrated my preferred share portfolio tool, [Sharesight](https://www.sharesight.com/), using a similar approach. See [sharesight-skill](https://github.com/lextoumbourou/sharesight-skill).

Dan Shapiro [^3] discusses 5 levels of software autonomy, with **Dark Software Factory** at Level 5, where software is a "black box that turns specs into software". I might not quite be operating at Shapiro's level 5 here - I still find myself butting in on the agent's work with my opinions about code quality - but I can certainly see the path towards it.

For my actual work at Canva, our users depend on us getting things exactly right, and currently, [engineers must own AI-generated code as if they wrote every line themselves](https://www.linkedin.com/posts/brendanhumphreys_no-you-wont-be-vibe-coding-your-way-to-activity-7305080254417547264-qidy/), so we're not doing Dark Software Factory anytime soon. But agentic coding is a fact of life. Even if the code itself is carefully peer-reviewed, there are definitely lessons to take away: how can I make sure the agent has all the context it needs? How can I enable it to validate at every stage of the implementation, and how can I provide feedback to guide its work? How can I be as exhaustive as possible in all the testing scenarios, removing any means of cheating?

The present of engineering is becoming more about reviewing code than writing it. But the future of engineering might be more about exhaustive verification of your system's correctness, and not much about the actual code at all.

Discussion on [Bluesky](https://bsky.app/profile/notesbylex.com/post/3mecieamjac2d), [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7426027007207059456/?originTrackingId=maBhvwgNnwJXZQ%2BUNUWKbA%3D%3D) or [Mastodon](https://fedi.notesbylex.com/@lex/116031765064843416).

*Cover by <a href="https://unsplash.com/@homaappliances?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Homa Appliances</a> on <a href="https://unsplash.com/photos/a-machine-that-is-inside-of-a-building-_XDK4naBbgw?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>*

## References

[^1]: Moynihan, L. (2024, December). [*The Software Factory*](https://lukepm.com/blog/the-software-factory). LukePM.com.
[^2]: StrongDM. (n.d.). [*StrongDM Software Factory*](https://factory.strongdm.ai/).
[^3]: Shapiro, D. (2026, January). [*The Five Levels: From Spicy Autocomplete to the Dark Factory*](https://www.danshapiro.com/blog/2026/01/the-five-levels-from-spicy-autocomplete-to-the-software-factory/).