---
category: essay
title: "6 Months of OpenClaw"
date: 2026-07-11 17:30
modified: 2026-07-11 17:20
status: draft
tags:
- OpenClaw
- Obsidian
---

(Note: this article is a work in progress if you somehow stumbled across it.)

I've been using OpenClaw for 6 months now.

OpenClaw was really popping off on the internet when it came out all those months ago, but there's certainly a lot less talk about it in my feeds.

But for me, it became an integral part of my life.

In this article, I wanted to share all the use cases that I still use it for.

## My High-Level Workflow

The original article I wrote goes into detail about my setup: [OpenClaw: the missing piece for Obsidian's second brain](openclaw-the-missing-piece-for-obsidians-second-brain.md).

I have an Obsidian vault - a collection of notes that I've been accumulating since 2020. I use the pattern of creating a daily note that links to other project journals and so forth.

The OpenClaw files are just a directory in the repo.

## Morning Routine - create Daily Note

I have a cron that runs every morning that creates the Daily Note around 4am.

It adds any key tasks that I need to remember from my todo list. It also logs any results from the stock market last night, and tracks my spending via the Pocketsmith Skill.

It tells me what the day's workout is going to be, based on the `workout/program.md` that I have defined.

The daily note is the default place for anything that happens in that day. I instruct OpenClaw to write to it whenever I tell it something to remember. However, my preference is to also have project files in one place. So if I have an ongoing project I will typically have either a journal file that I write to, or a journal section of the project page. Each time I update the journal, I will also update the daily note.

## Calorie tracking

Late last year, I started working towards a goal of getting lean. The only viable way to do that is to consume fewer calories than you burn, and the best way to do that is to track your calories.

I've found OpenClaw to be really handy for this. I typically weigh my food and tell Obsidian what I ate, or take a picture of the nutrition label, or, if all else fails, just take a picture of it.

I have a simple skill for this called `food-log`, which is instructed to first consult a `common-foods` file. This allows me to build up a collection of precise macros, especially for things that can be annoyingly hard to find online, like fast food items.

I used to use ChatGPT projects for this, but I prefer to be LLM agnostic wherever possible.

## Weight Tracking

Every morning, the first thing I do is weigh myself, naked after using the toilet.

When I give it my weight, it triggers the next part of the daily routine:

* Save my weight to frontmatter.
* Run a `weight-agg` script that tracks my weight over time.
* Fetch the sleep score from Oura.

You can see that my weight has progressed nicely:

![Weight Progress - All Time](../_media/6-months-of-openclaw/weight-all-time.png)

There are also some before-and-after photos, although I didn't think to create a great before shot.

![Before and after — Jan 2026 (91.5 kg) to Jul 2026 (81.4 kg)](../_media/6-months-of-openclaw/weight-before-after.jpg)

## Life Admin - A Place for everything

Having a chatbot that has access to my life admin files is handy.

I give OpenClaw read access to my emails, and it checks them periodically.

It's nice to just have a dumping ground for any upcoming dates to track.

If I need to do repairs around my house, I'll immediately create a new project file to keep track of any research I need to do.

Any notes I get from the vet get saved to my dog's file, and it gives me a sense of being organised.

## Skills

I find it somewhat useful to turn tasks into Skills. A lot of the time skills can be useful outside of the context of OpenClaw, so just packaging up a workflow into a skill means I can reuse it in Codex or Claude Code if I need to.

## Lessons

### Keep cron jobs light—reference skills

### It's not a coding agent
