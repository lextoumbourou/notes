---
category: essay
title: "6 Months of OpenClaw"
date: 2026-07-11 17:30
modified: 2026-07-12 10:17
summary: "Macros, workouts and life admin."
status: draft
tags:
- OpenClaw
- Obsidian
---

OpenClaw was really popping off on the tech internet when it came out earlier this year, but if Google Trends is anything to go by, the hype has died significantly.

<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/4448_RC01/embed_loader.js"></script>
<script type="text/javascript">
trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"openclaw","geo":"","time":"2025-07-11 2026-07-11"}],"category":0,"property":""}, {"exploreQuery":"q=openclaw&hl=en-GB&legacy&date=today 12-m","guestPath":"https://trends.google.com:443/trends/embed/"});
</script>


But it has become an integral part of my life over the last 6 months. It hasn't run a 5-figure SaaS business for me, or gone rogue posting sci-fi LLM slop on a message board, but it has proved to be a very convenient entry point into my Markdown-based life admin system.

I wrote an article about my setup a few weeks into my OpenClaw journey, and it hasn't changed much.

I still feel like it's the final piece to the promise of having a "second-brain" - the external memory store for everything going on in my life.

In this article, I want to do a kind of retrospective of what I've got out of running a personal OpenClaw instance, and share a few things I've learned.

## My High-Level Workflow

I have an Obsidian vault - a collection of Markdown notes that I've been accumulating for years. Every day, I create a "daily note" that lists my tasks and serves as a dumping ground for any notes and info I need to remember. I also create separate notes for projects, people and topics. Each of those notes has a journal, with entries that use the current date as a heading, linking back to the daily note.

I also maintain a centralised to-do list, and aim to put everything here, with links to the project pages.

Initially, my vault was maintained entirely by hand, but over time, it's become a hybrid of a [LLM Wiki](llm-wiki.md) and a journal. I still like to do a lot of writing by hand,  but I'm also okay with an LLM managing certain parts of the vault, especially admin stuff.

OpenClaw runs on a personal laptop that's always on. Its workspace is a folder in my vault, and the main OpenClaw files are simply pointers to existing files in my vault.

I also like the flexibility to use whichever agent I need - at work, I use both Claude Code and Codex, and for some personal projects, like tax filing, I like to use a coding agent. The vault can be used interchangeably by all.

OpenClaw also never sees anything relating to work. I maintain a separate vault for work stuff that only exists on my work machine. And I never run OpenClaw on my work machine (in fact, my work prohibits it).

## Calorie and weight tracker

Late last year, I set myself a goal of getting lean. The only viable way to do that is to consume fewer calories than you burn, and the best way to do that is to track your calories.

Generally, I found LLMs to be a convenient way to do this.

I've found OpenClaw to be really handy for this. I typically weigh my food and tell OpenClaw what I ate, or take a picture of the nutrition label, or, if all else fails, just take a picture of the meal.

I have a simple skill for this called `food-log`, which is instructed first to consult a `common-foods` file. The common foods file lets me build a collection of precise macros, especially for items that can be annoyingly hard to find online, like fast-food items.

I used to use ChatGPT projects for this, but I prefer to be LLM agnostic wherever possible.

## Weight Tracking

Every morning, the first thing I do is weigh myself, naked, after using the toilet.

When I send OpenClaw my weight, it triggers the next part of the daily routine:

* Save my weight to the frontmatter.
* Run a `weight-agg` script that tracks my weight over time.
* Fetch the sleep score from Oura.

You can see that my weight has progressed nicely:

![Weight Progress - All Time](../_media/6-months-of-openclaw/weight-all-time.png)

There are also some before-and-after photos, although I didn't think to create a great before shot.

<table style="width:100%; table-layout: fixed; border: none;">
  <tr>
    <td style="text-align:center; vertical-align:top; border: none;">
      <a href="../_media/6-months-of-openclaw/weight-before-jan-2026.jpg" target="_blank">
        <img src="../_media/6-months-of-openclaw/weight-before-jan-2026.jpg" style="max-width:100%;" alt="Before - Jan 2026 (91.5 kg)" />
      </a><br>
      <span style="font-size:smaller;">Jan 2026 - 91.5 kg</span>
    </td>
    <td style="text-align:center; vertical-align:top; border: none;">
      <a href="../_media/6-months-of-openclaw/weight-after-jul-2026.jpg" target="_blank">
        <img src="../_media/6-months-of-openclaw/weight-after-jul-2026.jpg" style="max-width:100%;" alt="After - Jul 2026 (81.4 kg)" />
      </a><br>
      <span style="font-size:smaller;">Jul 2026 - 81.4 kg</span>
    </td>
  </tr>
</table>


## Morning Routine - create Daily Note

I have a cron that runs every morning that creates the Daily Note around 4 am.

It adds any key tasks that I need to remember from my to-do list. It also logs last night's stock market results and tracks my spending via the Pocketsmith Skill.

It tells me what the day's workout will be, based on the `workout/program.md` I have defined.

It checks birthdays and any other important events.

The daily note is the default place for anything that happens on that day. I instruct OpenClaw to write to it whenever I tell it something to remember. However, I also prefer to have project files in one place. So if I have an ongoing project, I will typically have either a journal file I write in or a journal section on the project page. Each time I update the journal, I will also update the daily note.

## Life Admin - A Place for Everything

Having a chatbot that has access to my life admin files is handy.

I give OpenClaw read access to my emails, and it checks them periodically.

It's nice to just have a dumping ground for any upcoming dates to track.

If I need to do repairs around my house, I'll immediately create a new project file to keep track of any research I need to do.

Any notes I get from the vet get saved to my dog's file, and it gives me a sense of being organised.

## Email

It's helpful to grant OpenClaw access to your emails. But at the same time, just having it read everything that comes in burns through a lot of tokens, and LLMs aren't really discerning enough to figure out signal from noise.

My compromise is a heartbeat task that checks my inbox 2-3 times a day during business hours, flags anything important, and otherwise stays quiet.

## Skills

I find it somewhat useful to turn tasks into Skills. A lot of the time, skills are useful outside the context of OpenClaw, so packaging a workflow as a skill lets me reuse it in Codex or Claude Code if I need to.

## Models and Costs

I've managed to mostly get away with a basic Codex subscription. Sometimes I've needed to add extra credits to top up my monthly credit allocation, but not very often. It's fairly manageable.

The other lever is keeping `HEARTBEAT.md` small. It gets read on every poll, so every line in that file is a recurring token cost.

### Principles

- Everything I need to remember in my personal life goes in the vault. Documents, reminders, to-do items, and book recommendations - they all have one place.
- OpenClaw only gets read-only access to external services.
- LLM and agent agnostic. Everything is Markdown, and it's structured the way I want. I have CLAUDE.md that points to AGENT.md. My .agents/skills dir is symlinked in all the places that the agents expect.
- Keep cron jobs super light. Point the cron jobs to markdown files or skills.
