---
category: essay
title: "6 Months of OpenClaw"
date: 2026-07-11 17:30
modified: 2026-07-13 16:30
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

---

But I'm still using it every day, and have been ever since I set it up in late January, mainly as a convenient entry point into my Markdown-based life admin system.

For the unfamiliar: OpenClaw is a self-hosted AI agent that runs on my laptop. I message it via WhatsApp, and it can read and update my Obsidian vault, run scripts, and check a few external services on a schedule.

I still feel like it's the final piece to the promise of having a "second-brain"; it's the digital memory store for everything going on in my life.

The part I value most is that none of it is locked away with a vendor: OpenClaw's memory is my vault - Markdown in version control that I can read and edit. When a better LLM comes along, I can just switch, and my memory and system prompts come with me.

I wrote an article about my setup a few weeks into my OpenClaw journey, and my setup hasn't meaningfully changed much (see [OpenClaw: the missing piece for Obsidian's second brain](openclaw-the-missing-piece-for-obsidians-second-brain.md)).

In this article, I want to do a kind of retrospective of what I've got out of running a personal OpenClaw instance, and share a few things I've learned.

## My High-Level Workflow

I have an Obsidian vault - a collection of Markdown notes that I've been accumulating for years. Every day, I create a "daily note" to write my to-do list in, and it serves as a dumping ground for any notes and info I need to remember. I also create separate notes for projects, people and topics. Each of those notes has a journal, with entries that use the current date as a heading, linking back to the daily note.

I also maintain a centralised to-do list, and aim to put everything here, with links to the project pages.

Initially, my vault was maintained entirely by hand, but over time, it's become a hybrid of a [LLM Wiki](llm-wiki.md) and a journal. I still like to do a lot of writing by hand, but I'm also okay with an LLM managing certain parts of the vault, especially admin stuff.

OpenClaw runs on a personal laptop that's always on. Its workspace is a folder in my vault, and the main OpenClaw files are simply pointers to existing files in that folder.

I also like the flexibility to use whichever agent I need - at work, I use both Claude Code and Codex, and for some personal projects, like tax filing, I like to use a coding agent. The vault can be used interchangeably by all.

OpenClaw also never sees anything relating to work. I maintain a separate vault for work stuff that only exists on my work machine. And I never run OpenClaw on my work machine (in fact, my work prohibits it).

I use WhatsApp to communicate with OpenClaw, and bought a new SIM card to give me a separate WhatsApp account.

## Primary use cases

### 1. Calorie, weight and workout tracking

Late last year, I set myself a goal of getting lean, and doing so has meant tracking calories for every meal I eat and trying to stay in a deficit. Generally, I found LLMs to be a convenient way to do this. The freeform nature of the text and the fact that most of them can absorb photos and labels make it pretty straightforward to track meals.

For things that have labels, or where I know the calories (like fast food or restaurants that publish their nutrition info online), I'll tell OpenClaw the specific calories and also have it update a log of food I've eaten before, called `common-foods`.

For meals I cook, I'll typically give OpenClaw the recipe and then weigh the portions to get precise measures.

I aim to eat about 2200 kcal per day, which should be about a 500 kcal deficit per day at my weight and height of around 181cm.

Then, every morning, I weigh myself, naked, after relieving my bladder.

OpenClaw tracks my daily weight, and a script generates this nice, pretty graph.

As you can see, the trend is moving nicely, except for a few blips for holidays, where I didn't do any calorie counting and went back to my boozy old ways.

![Weight Progress - All Time](../_media/6-months-of-openclaw/weight-all-time.png)

I follow a muscle-building plan based on ideas from Jack Woods and Mindful Movers (who I highly recommend), which involves performing maximum-effort callisthenics movements a few times a week. I also track stretching and use it to track recovery from any injuries. Once a week, I share body progress photos, and OpenClaw saves them in a nice table.

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

### 2. Life Admin - everything goes here

On top of helping me with my fitness goals, it's simplified the way I manage different things in my life. Like preparing for my taxes, planning holidays, doing maintenance around the house, etc.

I have project files for all the different things going on in my life, and encourage OpenClaw to create new ones if needed.

Then, documents get stored in a `_media` directory and are linked to the project. If I donate some money to a fundraiser, I'll tell OpenClaw, and it will track it in my tax deductions. If I get invited on a trip, I'll tell OpenClaw, and it will add the items to my to-do list and create a project file.

If something needs to be done around the house, I'll create a new project file for it. Then, use it to track any research I do into tradespeople, and make sure to log any conversations I have.

When I have to take my dog to the vet, I'll log any notes from the vet and track any medication in my Obsidian to-do list.

Any reminders, like subscriptions to cancel, all go into the vault via OpenClaw.

There's just one place where everything goes: whether it's in Dropbox, in email, or sent by mail, I make sure it ends up in my vault, and anything I need to act on is turned into a to-do item.


### 3. Morning Routine

I have a cron that runs every morning that creates the Daily Note around 4 am. It checks my to-do list for any projects or items that need action. It scans my people files for any upcoming birthdays. I have it connected to my finance-tracking software [Pocketsmith](https://my.pocketsmith.com/), and it shows how my spending is going. It has access to my stock portfolio and tells me how it's doing.

It tells me what workout I'll be doing today.

Then, after I give it my morning weight, it knows that I'm up and fetches stats from Oura, like my sleep and how many steps I walked the night before.

Not life-changing stuff, just kinda handy.

Again, it's nice to have all this in one place. If I don't want to use Oura anymore, I can track my sleep some other way, but my data lives on.

## Principles and Lessons

### Tracking is the killer feature

The thing I find most appealing is how simple it makes tracking the things I actually care about. There is a useful principle in habit formation: what gets measured gets managed. The hard part is usually not creating a spreadsheet or a note. It is remembering to open it, finding the right place, and doing the boring little update every day.

With OpenClaw, I can send a message in the app I already use. Calories, morning weight, stretching and workouts all become data points in a system that can show me the trend later. The same applies to less numerical goals: I can track whether I have written social media posts this week, or keep a running journal for a project rather than relying on my memory.

It also works for the mundane stuff that would otherwise disappear. I currently have an intermittent tapping sound somewhere around the upstairs bathroom. Each observation goes into a project journal, with a link back to that day's note: where it was audible, whether it rained, and how long it was between sounds. None of this is individually impressive, but together it means small bits of evidence do not evaporate before they become useful.

### Everything in one place

Everything I need to remember in my personal life goes in the vault. Documents, reminders, to-do items, and book recommendations - they all have one place.

When I have other projects I'm working on, like family history or other open-source projects, I still tend to prefer to keep them in my vault or symlink them to it.

I'll keep a project journal in the vault and update it as I work on the project. Again - everything in one place.


### Read-only Access Only

I find it convenient to give OpenClaw access to a few things, particularly email, my finances, and my investments, but it's all read-only. I'm not comfortable having an agent do things on my behalf, and frankly, why would I want to? I like sending emails to people.

The vault itself is the exception - OpenClaw can write to it. But the whole vault is in git, so if it ever makes a mess of a file, recovery is one revert away. (The rest of my security setup is covered in [OpenClaw: the missing piece for Obsidian's second brain](openclaw-the-missing-piece-for-obsidians-second-brain.md).)

I use [googleworkspace/cli](https://github.com/googleworkspace/cli) for Gmail/Docs access; I use my [pocketsmith-skill](https://github.com/lextoumbourou/pocketsmith-skill) for Pocketsmith access; and [Sharesight Skill](https://github.com/lextoumbourou/sharesight-skill) for my investments.

### Keep Cron Jobs Light

Cron jobs get pretty unwieldy pretty fast, and they're not convenient to edit.

My rule is typically to keep cron jobs small, and they either point to a file or a skill that provides the instructions.

### Use a Real Agent When You Need to Do Real Work

Some bigger projects just do not make sense to do via OpenClaw - like coding, and even admin tasks like preparing my tax return for my accountant.

I'll use Codex or Claude Code in that case.

I try to make sure that my repo is totally agent-agnostic, and the OpenClaw stuff just points to files that other agent files, like AGENTS.md and CLAUDE.md, also point to.

## Models and Costs

*(All figures as of July 2026.)*

After a bit of exploring options, I've found that a $20 Codex subscription tends to be enough to do everything I need in OpenClaw. Sometimes I've needed to add extra credits to top up my monthly credit allocation, but not very often. It's fairly manageable.

I'm using between about 6 and 21 million tokens a day, but most of it is cache.

![OpenClaw Token Usage - Daily](../_media/6-months-of-openclaw/openclaw-tokens-daily.png)

My initial attempt at using Claude directly via the API had me racking up $10-20 in API costs per day. LOL. Eventually, I moved to GPT-5.4 as my main session model, with GPT-5.4-mini for heartbeats (OpenClaw's periodic background check-ins, rather than cron jobs that run at exact times).

Recently, I've cut over to the GPT-5.6 family of models, and it's another order of quality for around the same price - really nice.

I'm running `gpt-5.6-terra` for the main session, and `gpt-5.6-luna` handles heartbeats and cron jobs (terra and luna are OpenAI's mid and budget tiers of the 5.6 family).

## More phone time is a downside

One downside I want to mention is that I've become even more tied to my phone than before. Every meal, workout, and stretch needs my phone to record it. And phone time tends to lead to more phone time.

I have been working on having long stretches without my phone. I treat my morning coffee, any time I'm walking the dog, or time alone with my wife, as no-phone time.

With all the convenience that tech offers us comes the pain of having too much tech in our lives.

Also, it's a new software project, which means it's going to be broken a bunch. The OpenClaw development team is pretty quick to patch issues, though I must admit that the GitHub issue board is quite hard to follow with all the AI slop that's posted everywhere. Wish they had a policy of human-only issues, like other open-source projects.

## OpenClaw vs Hermes

Seems that during the time I've been experimenting with OpenClaw, a lot of people got excited by a similar project called Hermes, which promises to be a self-improving AI agent that "grows with you". I had a brief look at Hermes, but I decided that a self-evolving agent isn't really what I want.

I just want something that I can get working exactly the way I want, and that remains consistent.

Even in my own experimenting, any changes that I made to the system had unexpected knock-on effects that proved frustrating. Adding additional scripts and routines to the daily note sometimes caused it to fail. I love the new capability the LLMs have unlocked in the world, but I don't think they're quite ready to self-improve - more likely self-destruct, if left unchecked.

Hermes likely isn't for me - but if it works for you, that's great.

## Wrap Up

I've lost weight and feel like I look great. That's enough for me to be happy with my pal OpenClaw. I love it for life admin.

I haven't managed to run a 5-figure monthly SaaS business from it, and my OpenClaw hasn't gone rogue by posting spam on message boards, but it's a handy companion, and I think it's a worthy open-source project.