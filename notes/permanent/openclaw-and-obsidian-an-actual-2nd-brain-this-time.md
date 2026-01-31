---
title: "OpenClaw and Obsidian - an actual 2nd brain this time"
date: 2026-01-29 00:00
modified: 2026-01-29 11:40
summary: "how I integrate OpenClaw with Obsidian"
cover: /_media/openclaw-and-obsidian-cover.png
status: draft
tags:
- Obsidian
- OpenClaw
- Notetaking
---

I've been an Obsidian user for many years. I like it a lot. I like the paradigm of linking notes borrow from [Zettelkasten](zettelkasten.md). I like having a single place to keep all my notes, to track progress on my goals and so forth.

My workflow is pretty simple. I just create a new Daily Note every day, it has my todo list and also notes that I take during the day. Then, I also use notes to keep track of ideas and I link those to daily notes. I also create notes to help me learn and memorise new concepts. When I have a note that I feel is meaty enough, I'll make it public on my blog, so I kind have a system for blogging, albeit, I end up with this pretty odd blog, that's somewhere in between a blog and a half-baked personal Wiki. I also have folders for tracking people in my life (Birthdays and various things I want to remember about them), as well as using it for life admin tasks, like managing house repairs and storing recipes.

There are aspects of my life that end up outside of Obsidian. I have been using a ChatGPT project to track calories, for some fitness goals I'm working towards. I use Oura Ring for sleep and step tracking. Information pertaining to my house is scattered throughout emails and Dropbox, and in files. My wife and I use Google Sheets for finance tracking. I use Claude and Gemini for various projects, and a lot of useful information doesn't end up back in Obsidian.

Recently, like a lot of the tech-internet, I've been playing with OpenClaw (former Moltbot, former Clawdbot) and it really feels like the missing piece of the puzzle for Obsidian, in actually living up to the claim of being a "second brain".

Plus, OpenClaw is an actual useful agent - really truely feels like we're starting to see the promise of what AI can do for us. The major difference is that it's not just reactive (responding based on our prompts) but it's able to be proactive and actually useful.

## Basic Setup

I communicate with OpenClaw (who I refer to as **M**), via WhatsApp. I took the approach of creating a new WhatsApp account, by purchasing a $10 sim card from the supermarket in order to sign up for WhatsApp.
 
The OpenClaw workspace is where it stores its memory, and information about tools available to it, alongside it's [soul](https://docs.openclaw.ai/reference/templates/SOUL) file. By default, that workspace lives in `~/.openclaw/workspace` but I've put it in my Obsidian value under `~/private-notes/openclaw` so it's synced alongside my Obsidian files (I use [Obsidian Git](https://github.com/Vinzent03/obsidian-git) for syncing), and I can just update it like regular files.

I have my Obsidian vault explained to OpenClaw in my `TOOLS.md` file, which has been updated through a combination of conversations with OpenClaw, manual edits and Claude Code for major refactors. Here's the first few lines of it so you get a sense:

```TOOLS.md
# TOOLS.md - Local Notes

## Obsidian Vault

My workspace is `molt/` inside Lex's Obsidian vault. Access vault files via `../`.

- **Key files:**
  - `../todo.md` — current tasks with deadlines
  - `../finances/finances-summary.md` — property, stocks, crypto
  - `../health/health-summary.md` — health info and fitness routines
  - `../people/birthdays.md` — important birthdays
- **Daily notes:** `../daily/YYYY-MM-DD.md`
- **Subscriptions:** `../finances/subscriptions.md` — alert before renewals flagged for review
- **Projects:** `../projects/` and `../bachelors-degree-2022/cm3070-final-project/`
- **Sickness log:** `../health/conditions/sickness-log.md`

## Current Focus

**Source of truth:** `../todo.md`

Always read todo.md for current priorities, deadlines, and what's in progress. Don't maintain a separate list — todo.md is canonical.
# ... 
```

You don't need to have a perfect `TOOLS.md` file upfront - just work with OpenClaw to craft it the way you want.

I've been using OpenClaw to manage a few different aspects of my life.

## 1. Tracking health metrics

I have an Oura Ring, which I used for sleep and steps count tracking. I find it really useful, but I also like the idea that my data is mine to keep. If I ever find another way to track sleep or steps that I prefer, I'm free to move. So getting the data  into my existing Obsidian vault feels like a good idea.

A key feature of OpenClaw are skills which are basically some markdown and some scripts that give the OpenClaw (or any agent that supports them) some new capability.

There's already a great skill for the Oura ring, which I installed following the [instructions](https://github.com/kesslerio/oura-analytics-clawdbot-skill):

And I get M to fetch my sleep and steps on a heartbeat, and put it into my Daily Notes.

I also weigh myself each morning, and message OpenClaw to save that to my daily note, and I have aggregate notes that show me my progress overtime. I used to use DataView for this task, but I feel like it makes more sense to just have a script that generates a new file with the aggregate metric, so OpenClaw has an easy way to view my progress.

### 2. Workouts and Calorie tracking

When I workout, I share various metrics with OpenClaw, and encourage it to keep me consistent with things like flexibility routines and daily step goals.

I'm in a cut-phase of my health journey and I will take a picture, of what I eat and use it to track calories, aiming to stay under 2200 per day, until the cut finishes.

I have been using GitHub projects for this previously, and while it works well, I'd rather have the sense that I have total ownership over my data.
## 3. Morning Routing

I have a cron job that runs each morning every day to create my daily note. M reads my `todo.md`, finance summary, and health summary, fetches my Oura sleep data, and creates a note with suggestions for what I should focus on that day.

The note gets created with a "M Suggestions" section that prioritizes my tasks based on deadlines. When it's done, I get a WhatsApp message letting me know.

I keep a `birthdays.md` file in my Obsidian vault with important dates. M checks this each morning and alerts me if anyone's birthday is coming up. I also put appointments in `todo.md`

## 4. Projects Management and Todo list

My `todo.md` is the source of truth for what I'm working on. M reads it to understand my priorities and deadlines, and factors them into daily suggestions. I don't maintain a separate list anywhere, todo.md is canonical.

## 5. Personal CRM

I have a `people/` folder in my vault with a note for each person I want to remember things about - family, friends, colleagues. Each note has sections for present ideas, gifts I've given them, and a journal of interactions.

After a phone call with my dad, I'll message M with a quick summary - what we talked about, any updates on his life, things I want to remember. M appends it to his note with a link back to my daily note. When his birthday comes up, I can look at the present ideas section and see what I've given him in the past.

It's basically a personal CRM. Before OpenClaw, I'd forget to update these notes because the friction was too high. Now I just send a voice message after a call and it gets captured.

## 6. Property and Contractor Tracking

I have a `property/` folder with a subfolder for each property. Each one has a main file with the basics (mortgage, rates, insurance), plus a `repairs.md` for tracking maintenance.

The repairs file has a summary table with date, description, cost, contractor, and status. Below that, detailed notes for each job - before/after photos, invoice PDFs, and if I made an insurance claim, the claim number and reimbursement details.

When something breaks, I message M with a photo and description. It creates an entry in the repairs log and links it to my daily note. When the contractor comes out, I send the invoice and M adds it to the entry. If there's an insurance claim, same thing.

It's great for tax time - all the repair costs for investment properties are in one place with receipts attached. And when I need to call a tradie again, I can see who I've used before and whether they were any good.

## 7. Recipes

I have a `recipes/` folder with a note for each dish I make regularly. The current recipe sits at the top, and below that is a journal where I track modifications and what worked.

When I cook something and try a new technique, I message M with what I did and how it turned out. It adds a dated entry to the journal. If the modification was good, I update the main recipe to reference it. Over time, each recipe evolves based on actual results rather than just following the original blindly.

It's useful for things like browning meat properly or getting the ratio of stock to tomato paste right. I can look back and see what I tried last time and whether it was an improvement.

---

## Avoiding Analysis Paralysis

One of the pitfalls I've found myself falling into in the past was analysis paralysis. I know Obsidian could be useful for tracking so many things, but I'd get stuck trying to design the perfect system before actually using it.

I like to have everything together: insurance details, property repair logs, present ideas for family members. But I always found myself getting bogged down coming up with the "ultimate" system. Now I just tell the LLM what I want to achieve and have it come up with a plan, ensuring that it updates its internal memory with what it came up with. I find Claude Code a better tool for this job than OpenClaw - it's better at working with a lot of files at once.

---

It really has become an indispensable part of my life. The token costs are high, I don't even want to mention how much I've spend on Opus 4.5 tokens so far, but I think once the setup is done, I'll be spending a lot less per day.