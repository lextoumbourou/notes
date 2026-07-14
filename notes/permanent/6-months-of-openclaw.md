---
category: essay
title: "6 Months of OpenClaw"
date: 2026-07-15 08:30
modified: 2026-07-15 08:53
summary: "Macros, workouts and life admin."
tags:
- OpenClaw
- Obsidian
---

[OpenClaw](openclaw.md) went viral earlier this year on tech internet, but the hype seems to have died down quite a bit, especially if Google Trends is anything to go by.

![Google Trends: worldwide search interest for OpenClaw, July 2025 to July 2026](../_media/6-months-of-openclaw/openclaw-google-trends.png)

But I'm still using it every day, and have been ever since I set it up in January, when it was called Clawdbot.

I use it as an entry point to my Markdown-based (Obsidian) life admin and note-taking system. The combination of LLM agents, cron, and chat app connectors has proven very convenient.

I feel like it's the final piece to the promise of having a "second-brain". I also love that I'm building a [LLM Memory](memory.md) system that's vendor-agnostic; I can just switch to a different LLM vendor (ideally, one day, a locally running one) and all my memories switch with me.

I wrote an article about my setup a few weeks into my OpenClaw journey, and it hasn't changed meaningfully (see [OpenClaw: the missing piece for Obsidian's second brain](openclaw-the-missing-piece-for-obsidians-second-brain.md)). However, in this article, I want to reflect on what I actually achieved from running an OpenClaw instance and share a few things I've learned along the way.

## My High-Level Workflow

### Obsidian vault

I have a private Obsidian vault, which is a collection of Markdown notes and documents I've been accumulating for years. Every day, I create a daily note to plan my day, and I use it as a journal and a dumping ground for any notes and info I need to remember. I create separate notes for projects, people, trips and topics, which are linked back to the daily notes. Each of those notes has a journal that lets me put a thing-specific log of information in one place, and I link the journal entries back to the daily note, so I can retrieve information by day or by project. On top of that, I also maintain a centralised to-do list which tracks all my key tasks and deadlines. Initially, my vault was maintained entirely by hand, but in recent years, it's become a hybrid of an [LLM Wiki](llm-wiki.md) and a journal.

I still do all writing by hand, but I'm also okay with an LLM managing and updating certain files in the vault, especially projects that just need information dumps.

![My Obsidian vault: to-do list, daily note and a project note, all linked](../_media/6-months-of-openclaw/obsidian-workflow.png)

### OpenClaw and other agents

OpenClaw runs on a personal laptop that's always on. Its [agent workspace](https://docs.openclaw.ai/concepts/agent-workspace) is a folder in my vault, and OpenClaw instructions are mostly just to refer to existing files in my vault, instead of managing its own file structure. For any other agents I want to work with, such as Claude Code or Codex, I follow a similar pattern with their agent files. The goal is to have a vault which can easily be augmented by an LLM agent - but I'm not tied to any particular one.

I primarily use WhatsApp to communicate with OpenClaw. It's already the chat app that my family and friends use, and I bought a new SIM card to give me a separate WhatsApp account for my OpenClaw agent, which I've named **M**.

Also, I do use Obsidian and an LLM wiki for work, but I keep it in a totally separate vault that doesn't sync with Obsidian and that OpenClaw never sees. Work forbids us from using OpenClaw for work stuff for good reason.

## The main use cases

### 1. Calorie, weight and workout tracking

Last year, I set a goal of getting lean, and I have been trying to maintain a calorie deficit by tracking the calories in every meal I eat. Generally, I found LLMs to be a convenient way to do this - the fact that most of them can handle information via freeform text and/or images means they make it pretty straightforward to track meals, either via a specific weight and description, or via a food label or via an image of the food, if all else fails.

If I expect to eat the meal again and know the exact calories, I'll ask OpenClaw to log it in a file called `common-foods`, which it checks before estimating any food.

I'm aiming to eat about 2200 kcal per day, which should be about a 500 kcal deficit per day at my height of around 181 cm and current weight.

![How food tracking works: a WhatsApp photo, checked against common-foods.md, logged to the daily note's Macros table](../_media/6-months-of-openclaw/openclaw-food-log.png)

Then, every morning, I weigh myself naked after relieving my bladder, which gives me a consistent daily weight, and share it with OpenClaw, kicking off the second phase of my daily routine.

![Weight Progress](../_media/6-months-of-openclaw/weight-all-time.png)

As you can see, the trend is moving nicely, except for a few blips during holidays, when I didn't do any calorie counting and went back to my careless, boozy old ways.

For workouts, I follow a muscle-building plan based on ideas from [Jack Woods](https://jackhwoods.com/) and [Mindful Mover](https://mindfulmover.com/), which involves short, maximum-effort callisthenics sessions a few times a week. I track these with OpenClaw.

I am also trying to get more flexible, so I track my stretching and recovery from any injuries. Once a week, I share body progress photos, and OpenClaw saves them in a nice table.

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

What I've achieved here is straightforward: I've lost weight, my fitness is improving, and I have a clear record of the work that got me there.

### 2. Morning routines

I have a cron that runs every morning around 4 am to create the Daily Note. It checks my to-do list for any projects or items that need action. It scans my people files for any upcoming birthdays. I have it connected to my finance-tracking software [PocketSmith](https://my.pocketsmith.com/), and it shows how my spending is going. It has access to my stock portfolio and tells me how it's doing.

It checks my workout and stretching history, and tells me what I'll be working on today. I also have it fetch my OpenClaw token usage data, so I know whether I'll need to buy extra credits this month.

Then, after I give it my morning weight, it knows I'm up and fetches my sleep data from Oura, including how much I slept and how many steps I walked the day before.

I like that my Oura data is backed up into my vault. It gives me a sense of freedom from one particular vendor: if I want to track my sleep some other way, maybe a watch or something, my historic data lives on.

I'm proud to say I actually remember people's birthdays now and generally feel on top of things.

### 3. Life admin - a place for everything digital

On top of helping me with my fitness goals, it's simplified the way I manage things like preparing for my taxes, planning holidays and doing maintenance around the house.

I have project files for all the different things going on in my life, and I encourage OpenClaw to create new ones if needed.

Documents get stored in a media directory and linked to the project. If I donate some money to a fundraiser, I'll tell OpenClaw, and it will track it in my tax deductions. If I get invited on a trip, I'll tell OpenClaw, and it will add the items to my to-do list and create a travel-specific project file.

If something needs to be done around the house, I'll create a new house project file for it. Then I use it to track any research I do on tradespeople and to log any conversations I have.

When I have to take my dog to the vet, I'll log any notes from the vet and track any medication in my Obsidian to-do list.

Any reminders, such as subscriptions to cancel, go into the vault via OpenClaw.

There's just one place where everything goes: whether it's in Dropbox, in email, or sent by mail, I make sure it ends up in my vault, and anything I need to act on is turned into a to-do item. I give OpenClaw read-only access to my email so it can help me find and triage things, but it doesn't send anything on my behalf.

On top of calories and workouts, I tell M when I get sick, and it tracks my sicknesses, which can be handy for spotting patterns. If I notice a growth on my dog's body that the vet tells me to keep an eye on, I get M to create a weekly cron, and I take photos when I get alerted to show my vet.

There are tapping sounds around the house that I'm trying to get on top of. Birthday present ideas for my wife. Things I want to buy on my next trip to the supermarket. Subscriptions to cancel.

These things end up in daily notes, project files and people files, and I get reminders when I need them.

The result is that I don't lose track of documents, tradespeople, appointments, or the small life-admin jobs that otherwise quietly evaporate.

## Principles and Lessons

### Read-only access outside of the workspace

I find it convenient to give OpenClaw access to a few things, particularly email, my finances, and my investments, but it's all read-only. I'm not comfortable having an agent do things on my behalf, and frankly, why would I want to? I like sending emails to people.

OpenClaw can, of course, write to the vault, but it's all in Git, and I review the changes it makes periodically when I commit. The rest of my security setup is covered in [OpenClaw: the missing piece for Obsidian's second brain](openclaw-the-missing-piece-for-obsidians-second-brain.md).

I use [googleworkspace/cli](https://github.com/googleworkspace/cli) for Gmail/Docs access - a really handy way to turn emails into project journal items and to-dos; I use my [pocketsmith-skill](https://github.com/lextoumbourou/pocketsmith-skill) for PocketSmith access; and [Sharesight Skill](https://github.com/lextoumbourou/sharesight-skill) for my investments.

### Keep cron jobs light - point them to docs and skills

Cron jobs get pretty unwieldy pretty fast, and they're not convenient to edit.

My rule is typically to keep cron jobs small, and they either point to a file or a skill that provides the instructions.

### Use a coding agent for big jobs

Some bigger projects just do not make sense to do via OpenClaw - like coding, and even admin tasks like preparing my tax return for my accountant.

I'll use Codex or Claude Code in that case.

### Keep the vault agent-agnostic

Since I want to use the right agent for the problem, I try to make sure my repo is fully agent-agnostic. The vault explains itself in a root `AGENTS.md` file. Codex reads that directly; Claude Code imports it through `CLAUDE.md`; and OpenClaw's own agent file points back to the same instructions.

Personal facts live in normal notes, with `people/me.md` as my canonical profile. Reusable workflows, like food logging and my morning routine, live as shared [Agent Skills](https://github.com/lextoumbourou/lex-skills) under `.agents/skills`. Each agent can have a small amount of agent-specific configuration, but I don't want separate copies of how my vault works because they'll inevitably drift.

That means OpenClaw is just one interface to the system. If I decide to switch to something else, the notes, memories and workflows don't have to move with it.

## Models and Costs

(All figures as of July 2026.)

After exploring a few options and racking up some pretty painful Claude API bills, I've found that running GPT-5.6 through Codex on my $20 ChatGPT Plus subscription is usually enough to do everything I need in OpenClaw. Sometimes I've needed to add extra credits to top up my monthly credit allocation, but not very often.

I'm using between 6 and 21 million tokens a day, but most of it is cache.

![OpenClaw Token Usage - Daily](../_media/6-months-of-openclaw/openclaw-tokens-daily.png)

That kind of traffic on Opus 4.8 would cost $10-40 per day, depending on the cache mix, so you really need a subscription to make this work.

I had been using GPT-5.4 as my main session model, with GPT-5.4-mini - not the smartest model - handling "heartbeats", OpenClaw's background check-in process. But I've recently cut over to the GPT-5.6 family of models, and it's another order of quality for around the same price - they're really nice.

I'm running [`gpt-5.6-terra`](https://developers.openai.com/api/docs/models/gpt-5.6-terra) (the mid-tier model - think Claude Sonnet) for the main session, and [`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna) (the lower tier - think Haiku) handles heartbeats and cron jobs.

## OpenClaw vs Hermes

During the time I've been experimenting with OpenClaw, many people have got excited about a similar project called Hermes, which promises to be a self-improving AI agent that "grows with you". I took a brief look at Hermes and even tried installing it, which failed (maybe just because of laziness), but I decided that a self-evolving agent isn't really what I want.

I just want something that I can get working exactly the way I want and that keeps working in perpetuity.

Even in my own experiments, any changes that I made to the system had unexpected knock-on effects that proved frustrating. Adding additional scripts and routines to the daily note sometimes caused it to fail. I love the new capability the LLMs have unlocked in the world, but I don't think they're quite ready to self-improve - they're more likely to self-destruct if left unchecked.

Hermes likely isn't for me - but if it works for you, that's great.

## Complaints

I will say that one complaint is that I'm now annoyingly tied to my phone - even more so than before. But that's the price of tracking things meticulously, I guess.

Also, it's a new software project, which means it's going to be broken a bunch. The OpenClaw development team is pretty quick to patch issues, though I must admit that the GitHub issue board is quite hard to follow with all the AI slop that's posted everywhere. I wish they had a policy of human-only issues, like other open-source projects.

## Summary

I've lost weight and am happy with how my fitness is tracking; I never lose track of documents or tradespeople's phone numbers; I remember birthdays, etc. OpenClaw has been a welcome addition to my life.

I haven't managed to run a five-figure monthly SaaS business from it, and my OpenClaw hasn't gone rogue by posting spam on message boards, but it's a handy companion, and I think it's a worthy open-source project.
