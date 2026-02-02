---
title: Spec-First LLM Development
date: 2026-01-19 00:00
modified: 2026-01-19 00:00
summary: "in which the LLM maintains a spec file alongside the project"
cover: /_media/spec-first-llm-development-cover.png
tags:
- SoftwareEngineering
- AIAssistedSoftwareEngineering
---

**Spec-First LLM Development** is the simple idea that, instead of asking an LLM to immediately output code after prompting, you first ask it to output a spec file, which is continually updated as you and the LLM develop software together. The spec file is stored in version control and should be re-read by the LLM whenever new context is added.

Luke Bechtel writes a really useful article called [Vibe Specs: Vibe Coding That Actually Works](https://lukebechtel.com/blog/vibe-speccing) that has more details about how it looks in practice. Also, in a [Hacker News thread](https://news.ycombinator.com/item?id=46670279), antirez, creator of Redis, shared his approach, using [IMPLEMENTATION_NOTES.md](https://github.com/antirez/flux2.c/blob/main/IMPLEMENTATION_NOTES.md) in his flux2.c project. He says: *"This project was possible only once I started to tell Opus that it needed to take a file with all the implementation notes, and also accumulating all the things we discovered during the development process. And also, the file had clear instructions to be taken updated, and to be processed ASAP after context compaction. This kinda enabled Opus to do such a big coding task in a reasonable amount of time without losing track"*

Another really interesting solution to this problem comes from [Steve Yegge](https://en.wikipedia.org/wiki/Steve_Yegge) in his project [beads](https://github.com/steveyegge/beads), which is a much more engineered approach that builds a dependency-aware graph of tasks. It looks like Claude is going to implement a similar solution in Claude Code called [Tasks](https://x.com/trq212/status/2014480496013803643).

It's nothing new. The concept of [Software Requirements Specifications (SRS)](software-requirements-specifications-srs.md) was commonplace as early as 1975 [^1]. However, I suspect I haven't seen many of these in my career because keeping them up to date with the pace of software change was just too difficult an undertaking. Now that we have LLMs to do it for us, the story is very different.

It seems that a lot of the paradigms for writing high-quality software, that in practice often get pushed aside due to them being deemed too labour intensive - SRS being one obvious example, but other practices like [Test-Driven Development](test-driven-development.md) and [Property-based Testing](property-based-testing.md) - start to sound a lot more sensible when it's the LLM doing the hard labour, and the engineer reaping the rewards in the form of reliable software.

Personally, I've had some success adding implementation notes to my toolkit for recent projects. I've been working on porting some PyTorch projects to the MLX framework (see [mlx-demucs](https://github.com/lextoumbourou/mlx-demucs), [mlx-contentvec](https://github.com/lextoumbourou/mlx-contentvec) and [mlx-rvc](https://github.com/lextoumbourou/mlx-rvc) for recent examples), and anecdotally, it seems to prevent the LLM from getting stuck in loops, especially across context compactions. It also seems to help prevent it from making the same kinds of errors, like failing to use uv or using the wrong Python version for comparison testing, working better than instructions in [AGENTS.md](https://agents.md) or [CLAUDE.md](https://code.claude.com/docs/en/memory) instructions alone.

That said, a text file feels like a pretty rudimentary solution to [Memory](memory.md) for an LLM - I'm sure there's going to be a lot more exploration in this space.

---

Discussion on [Bluesky](https://bsky.app/profile/notesbylex.com/post/3mdqtpk6e5s2c, [Mastodon](https://fedi.notesbylex.com/@lex/115991998235821477) and [Mastodon](https://fedi.notesbylex.com/@lex/115991998235821477).

## References

[^1]: Ramamoorthy, C. V., & Ho, S. F. (1975). Testing large software with automated software evaluation systems. ACM SIGPLAN Notices, 10(6), 382–394. https://doi.org/10.1145/390016.808461
