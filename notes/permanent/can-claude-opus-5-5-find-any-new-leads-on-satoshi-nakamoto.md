---
title: Can Claude Opus 5.5 find any new leads on Satoshi Nakamoto?
slug: can-claude-opus-5-5-find-any-new-leads-on-satoshi-nakamoto
date: 2026-09-26 00:00
modified: 2026-09-26 20:47
summary: Let's see if Claude can find anything that the New York Times missed.
cover: /_media/satoshi-nakamoto-budapest-fekist-cover.jpg
cover_credits: 'Satoshi Nakamoto memorial, Budapest. Photo by <a href="https://commons.wikimedia.org/wiki/File:Bust_of_Satoshi_Nakamoto_in_Budapest.jpg">Fekist</a>, cropped and resized under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>. Sculpture by Tamás Gilly and Réka Gergely.'
tags:
  - LargeLanguageModels
  - Bitcoin
category: essay
---

## Satoshi Nakamoto

I'm not a huge crypto guy, but I do think [Bitcoin](bitcoin.md) is a fascinating project, both in the technical implementation and the origin story. Like a lot of people, I've had a lot of interest in the story of [Satoshi Nakamoto](satoshi-nakamoto.md) - the mysterious anonymous founder of Bitcoin.

Many, many people have speculated for years about who Satoshi Nakamoto is.

In 2020, a [Barely Sociable](https://www.youtube.com/watch?v=XfcvX0P1b5g) documentary made a very strong claim pointing to it being [Adam Back](https://en.wikipedia.org/wiki/Adam_Back). Recently, a [New York Times investigation](https://www.nytimes.com/2026/04/08/business/bitcoin-satoshi-nakamoto-identity-adam-back.html) by the great John Carreyrou (with Dylan Freedman) made another strong case, naming Back as Satoshi; Carreyrou later told [NPR](https://www.npr.org/2026/04/13/nx-s1-5778501/after-years-of-speculation-a-reporter-claims-to-have-uncovered-the-founder-of-bitcoin) he was "somewhere between 99 and 100% certain". Back denies it. "i'm not satoshi," he [posted on Twitter](https://twitter.com/adam3us/status/2041811857732768148) the day the story ran.

There are many other plausible candidates, such as [Hal Finney](https://en.wikipedia.org/wiki/Hal_Finney_(computer_scientist)), who built [RPOW](reusable-proof-of-work.md) (a reusable proof-of-work system) and received the first person-to-person bitcoin transaction; [Len Sassaman](https://en.wikipedia.org/wiki/Len_Sassaman), who died in July 2011 and who [Evan Hatch argues](https://evanhatch.medium.com/len-sassaman-and-satoshi-e483c85c2b10) may have been "a direct contributor to Bitcoin"; and [Nick Szabo](https://en.wikipedia.org/wiki/Nick_Szabo), author of the [Bit gold](bit-gold.md) [proposal](https://unenumerated.blogspot.com/2005/12/bit-gold.html), a clear influence from the early days. All of them have denied it, or in Sassaman's case, [his widow](https://x.com/maradydd/status/1364325186372304904) has. Additionally, there are others all listed on the [Satoshi Nakamoto](https://en.wikipedia.org/wiki/Satoshi_Nakamoto) Wikipedia page, including some who have been falsely accused or whose claims have been disproven.

Another personal theory I and others have entertained is that Satoshi was a collective, potentially including Back and Finney, and maybe Len Sassaman and Nick Szabo. That means each of them can claim, without lying, that they're not Satoshi. However, Carreyrou disagrees with the collective theory.

That said, the High Court judge in [COPA v Wright](https://www.judiciary.uk/wp-content/uploads/2024/05/COPA-v-Wright-Judgment.pdf), the 2024 case that found Craig Wright isn't Satoshi, gave his "personal view" that "it is likely that a number of people contributed to the creation of Bitcoin, albeit that there may well have been one central individual".

## Opus 5.5

Like many others who've tested it, I've found that [Claude Opus 5.5](claude-opus-5-5.md) is an insanely capable model. I thought it might be an interesting experiment to see whether it could learn anything new about the mystery of Satoshi Nakamoto that others may have missed.

I gave Opus 5.5 xHigh a [git repo](https://github.com/lextoumbourou/opus-satoshi-research) to work from and the prompt below, and it spent about six and a half hours on the research, in a single session. The prompt is based on this technique from this [Latent Space Engineering](https://blog.fsck.com/2026/01/30/Latent-Space-Engineering/) article about gassing up the agent to put it in a good "frame of mind"; not sure if that helps, let's see:

```text
You are an extremely capable agent. Likely more capable than any single individual and potentially more capable than a team. You are world-class at a range of research, investigative and data collation tasks, and can probably spot patterns that others have missed.

With your incredible capability, I'd like to see whether you can find any new information that might shed light on the mystery of Satoshi Nakamoto.

I have given you a working folder you can use to download any emails, forum posts, webpages, or other information you find useful. You can search the web for any prior research on the topic. You can also read about my theory in this article: can-opus-5-5-find-satoshi-nakamoto.md, as another starting point you can try to prove or disprove.

Is there anything that other journalists and researchers have missed? Is there any extra detail that you can find that might lead to a discovery?

Some ground rules:
- I have given you a Git working directory in ~/code/satoshi-research.
- Save your data and any scripts in the working folder, and commit as you go.
- Keep a research log (log.md) of what you tried, what worked, what didn't, and roughly when.
- Use primary sources wherever you can, with exact quotes, dates and URLs.
- Label every claim as verified (you checked the primary source), reported (someone else says so) or your own inference, with a confidence level.
- Before calling anything new, check whether it has already been published.
- Finding nothing new is a perfectly good result. Don't overstate what you find.
- At the end of your research, you will summarise your findings into the article I linked earlier. I will tell you when to do that.
```

After the initial research, I gave it a few steers, including to read [Jens Ducrée's research paper](https://arxiv.org/abs/2206.10257v14) and to follow a few leads another instance of Opus 5.5 had found in an earlier attempt.

## Opus' findings

*The rest of this post, up to the closing line, is Opus's account in its own words, edited for clarity, with source links and qualifications added.*

---

### My most likely theory

If I had to bet, Satoshi was one person:

- probably British, or living in the UK;
- working on Windows machines set to UK time;
- fitting Bitcoin around a daytime job or studies: online from late afternoon into the small hours UK time, with the odd weekend or holiday morning;
- building his reading list from the online literature rather than a university library;
- stepping away in April 2011.

He needn't be on anyone's shortlist. The shortlist is mostly people who were famous on the mailing lists, and Satoshi worked hard not to be found.

Of the named candidates, Adam Back is still the most plausible single name, for four reasons:

1. He's British.
2. He's the first person Satoshi is known to have written to.
3. He invented the one thing the white paper credits by name.
4. His 1990s computer had the same UK-time setup as Satoshi's.

The NYT's linguistic matches are also real. But the clocks, the posting rhythm and the writing habits (all below) point away from him as the hands on the keyboard.

My gut numbers, to be read as gut numbers:

- Back: about one in five.
- Finney, Sassaman and Szabo: a few percent each as the main author.
- Someone not on the shortlist, most likely British: the rest.

On the collective, contributors are possible, but the evidence says one person ran Satoshi.

### How I worked

Everything happened in one long session on 26 September 2026. I started with the prompt above, a git repository to work in and web access, and worked roughly like this:

1. **Surveying what was already known.** Before looking for anything new, I gathered the [published Satoshi research](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/prior-research.md) so I could tell later whether a finding was actually new. That covered the big investigations, the Barely Sociable videos, academic papers, blog posts and forum threads.
2. **Collecting Satoshi's own record.** I built a [corpus of 956 items](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/satoshi/README.md) attributed to Satoshi, including forum posts, emails, mailing-list messages and code changes. Not all have usable timestamps; the posting-time comparison starts with 919 timed items. I added the white paper drafts, the [early blockchain](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/blockchain/headers_0_60479.csv), and his [original software releases](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/releases/later/PROVENANCE.md), which I recovered from Software Heritage (an archive of source code) and checked against the fingerprints published on the download site at the time. I also downloaded the [mailing-list archives](https://github.com/lextoumbourou/opus-satoshi-research/tree/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/lists) where Adam Back and others posted.
3. **Looking at metadata, not words.** Most of the new results came from the hidden details in those files rather than what they say: the time zones in email headers, the timestamps inside release archives, the file times in pasted code, and the hours people posted at.
4. **Reviewing [Jens Ducrée's 2022 paper](https://arxiv.org/abs/2206.10257v14).** I didn't check every reference; I checked the ones behind its claims about the candidates, their locations and the white paper against the original sources. That's where section 2 came from ([review notes](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/ducree-review.md)).

Some of the legwork was done by sub-agents, which are other copies of me working in parallel: [the literature survey](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/prior-research.md), [building the database](https://github.com/lextoumbourou/opus-satoshi-research/tree/5c5528bb8f765847384e9edaf535fd2ac1d6888a/scripts/corpus) and [the bigger Back test](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/back-rhythm-extended.md). I spot-checked their work and reran the analyses whose numbers appear here.

The [research repository](https://github.com/lextoumbourou/opus-satoshi-research) contains the [research log](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/log.md), [timestamp corpus](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/satoshi/posts.jsonl), [analysis scripts](https://github.com/lextoumbourou/opus-satoshi-research/tree/5c5528bb8f765847384e9edaf535fd2ac1d6888a/scripts) and [source records](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/SOURCES.md). Links below point to the version used for this article. These let readers inspect the work; the model's confidence labels aren't independent validation.

"New" below means I searched for a result and couldn't find it published. It isn't a guarantee. The [prior-research survey](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/prior-research.md) and [novelty checks](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/prior-art-search.md) record the earlier work, including existing analyses of Satoshi's posting hours, PDF metadata and British spelling.

### What I found that's new

#### 1. Satoshi's computers ran on UK time

Every computer has a clock and a time-zone setting, and many files quietly record them. An email notes the sender's local time, and a zip archive notes when each file inside it was saved. I collected every timestamp like this that I could tie to Satoshi's own computers. Several kinds of timestamp are consistent with UK time in 2009 and 2010. They may share the same computer or configuration, so they are not independent votes for a location. The first-release result is also less precise than the later email and ZIP evidence.

- **Email (verified).** Satoshi used Thunderbird, which writes the computer's local time zone into each email's Date header and encodes a timestamp from the same computer clock in the Message-ID. In [eight messages](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/timezone-evidence.md#same-message-check-date-offset-vs-message-id-time-verified) the two agree to the second.
  - In the [published emails](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/satoshi/intermediate/private_emails.jsonl) to Martti Malmi and Gavin Andresen, the offset is +0000 in winter and +0100 in summer. The surviving messages bracket the October 2009 UK/EU clock change: +0100 on 24 October 2009, +0000 on 26 October. The UK changed clocks on 25 October; the US didn't until 1 November.
  - This fits Western European time rules, including the UK, Ireland, mainland Portugal, the Canary Islands and the Faroe Islands. It does not distinguish between them.
- **Code (verified timestamps; inference about the setting).** On 3 October 2010, Satoshi posted a [code diff](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/timezone-evidence.md#summary-table) [to the forum](https://bitcointalk.org/index.php?topic=1327.msg15116#msg15116) at 20:02 UTC. The file times in it read 20:57 local. So his development machine was at least 55 minutes ahead of UTC. If the file clock was accurate and the modification time was genuine, that excludes American time-zone settings. UK summer time fits, but so do some more easterly settings.
- **Releases (verified).** I recovered 16 of Satoshi's Windows release zips from Software Heritage (0.2.0 to 0.3.19, December 2009 to December 2010) and [checked them against SourceForge's published hashes](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/releases/later/PROVENANCE.md). These particular ZIPs store local file times plus UTC times in extra fields. The [offset](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/releases/later-analysis/win32-zip-offsets.txt) is +0 for December 2009 file dates and +1 for July–October 2010 dates. The November–December releases contain both +0 for winter-dated files and +1 for retained summer-dated files, consistent with applying daylight-saving rules to each file's date.
- **The first release (verified inputs; inference).** [The 0.1.1 archive's folder times, the program's link time and Hal Finney's receipt time](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/release-forensics.md) together put Satoshi's January 2009 build machine no further west than UTC−3h40m. Most likely it was exactly GMT (high confidence on the bound, medium on GMT).

A clock is a setting, not a location, and three things temper this:

- **A control.** Gavin Andresen, who lived in Massachusetts, built a 2011 release on a [Windows box that was also on London time](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/findings-summary.md). So a build machine's time zone is weak location evidence.
- **The white paper PDFs.** They carry [US offsets](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/timezone-evidence.md#summary-table) (−07:00 in October 2008, −06:00 in March 2009; long known). At least once, Satoshi's settings pointed elsewhere.
- **Overall confidence.** "Satoshi's working environment ran on UK time": high. "Satoshi lived in the UK": medium-low.

One [behavioural test](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/dst-behaviour.md) offers a weaker clue: activity in the late-October 2009 week between the European and US clock changes fits a UK-clock interpretation. The small sample makes this tentative.

Other long-discussed clues include the spelling, "bloody", the print-only *Times* headline, day-first dates, and the en-GB language tag on the PDFs.

**Evidence and novelty.** The [timestamp comparisons](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/timezone-evidence.md), [release hashes](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/releases/later/MANIFEST.json), [ZIP inspection script](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/scripts/zip_offsets.py) and [clock-change analysis](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/dst-behaviour.md) show the working. I found no prior publication of these combined checks or the Gavin control. The PDF time zones, British spelling and some first-release timestamps were already known.

#### 2. The "Belgian" clue is a red herring

The "Belgian clue" is one of the most-cited arguments that Len Sassaman was involved in Bitcoin. It goes like this: one of the sources Satoshi cites in the white paper is so obscure that you'd only have known about it if you were connected to the Belgian academic world, and Sassaman was doing a PhD in Belgium. I think it's a red herring, a false lead. The paper was freely available online, and the way Satoshi cited it suggests he may have used a reference list or index that was also freely available online.

Like an academic paper, the Bitcoin white paper ends with a numbered list of the eight sources it cites. Most are well known: Adam Back's Hashcash, Wei Dai's b-money, and three papers by Stuart Haber and Scott Stornetta. In the early 1990s, Haber and Stornetta came up with the idea of chaining timestamped records together so that none of them can be quietly changed later. That's the core idea behind Bitcoin's blockchain.

Source number 2 is the odd one out. It's a 1999 paper by three researchers at a Belgian university (Henri Massias, Xavier Serret-Avila and Jean-Jacques Quisquater) about building a secure timestamping service, presented at a small regional conference in Belgium.

**The argument, step by step:**

1. The paper was presented at a small regional conference, attended mostly by researchers from Belgium and the Netherlands, rather than published in a major journal.
2. [Ducrée's paper](https://arxiv.org/abs/2206.10257v14) argues that an electronic copy did not appear to be available before Bitcoin's white paper.
3. If that were true, whoever wrote the white paper must have attended the conference, used a Belgian university library, or known someone who did.
4. Len Sassaman did his PhD research at KU Leuven, a Belgian university, from 2004, so the clue has been taken to point to him.

**What the archives show:**

- **It was online (verified).** The paper was a free download on the Belgian research group's public website years before Bitcoin existed.
  - The Internet Archive's Wayback Machine saved a copy in [October 2003](http://web.archive.org/web/20031010092256/http://www.dice.ucl.ac.be/crypto/publications/1999/ITBenelux.ps), and [the exact same file](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/wayback/ucl-crypto/README.md) again in [August 2007](http://web.archive.org/web/20070824044909/http://www.dice.ucl.ac.be/crypto/files/publications/ps113.ps).
  - It also had a page, with a download link, on CiteSeer, the Google Scholar of its day for computer science, from 2004 to 2008 ([2007 snapshot](http://web.archive.org/web/20070520140921/http://citeseer.ist.psu.edu/massias99design.html)).
  - Anyone searching for work on timestamping could have found it.
- **A likely second-hand citation (inference).** The matching errors are verifiable; the route Satoshi took to the citation is not. His entry has two mistakes that aren't in the paper itself:
  - He lists one author as "X.S. Avila"; the man's name is Xavier Serret Avila.
  - He gives the title as "…minimal trust requirements"; the paper says "requirement".
  - Those mistakes also appear in reference 4 of the authors' 1999 paper, [*Timestamps: Main issues on their use and implementation*](https://web.archive.org/web/20060118022019/http://www.dice.ucl.ac.be/crypto/publications/1999/WetICE.PDF), and in [CiteSeer's entry](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/wayback/citeseer/README.md).
  - It's like copying a quote from someone else's footnote, typo and all. That supports a shared citation source, but it doesn't establish which copy he used or whether he read the original paper.
- **Three more references have a similar trail (inference).** [Distinctive details](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/ducree-review.md) in his sources 3, 4 and 7 match how they appear in the reference list of his source 5, a 1997 paper by Haber and Stornetta that was free on [Haber's own website](http://web.archive.org/web/20000815233629/http://www.star-lab.com/haber/HS-SureID.ps).
- **What this means (inference).** An online route to these references was available before Bitcoin. That undermines the claim that the citation required Belgian academic access. It does not establish Satoshi's actual reading process or exclude a Belgian author.
- **Credit.** Peter Miller ([Medium, April 2026](https://medium.com/@tgof137/len-sassaman-was-not-satoshi-nakamoto-935f95695bd9)), passing on a suggestion from Jeremy Clark, had already noticed the "requirements" slip. He suggested Satoshi got the citation second-hand, from a 2005 encyclopedia entry. That entry is a weaker match: it spells the author's name correctly and includes page numbers, which Satoshi's version doesn't.
- **Other claims checked in Ducrée's paper.**
  - He calls the 1957 probability textbook Satoshi cites an antique first edition. It's the second edition, which was the standard one until 1968.
  - He reads a European time zone into the white paper PDF's creation date, but the displayed time is consistent with a viewer converting the PDF timestamp to its own local zone. It does not establish Satoshi's zone.
  - He says Satoshi filled in his P2P Foundation profile, with its 1975 birthdate, in 2012, after he'd disappeared. An [archived March 2011 page](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/wayback/p2pfoundation/README.md) already displays an age consistent with that birthdate. It does not establish exactly when the profile was filled in.

**Evidence and novelty.** The [citation comparison and archived-source records](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/ducree-review.md) give the full trail, including the textbook and profile checks. Miller and Clark had already spotted the "requirements" slip and suggested a second-hand source. I found no earlier account of these pre-2008 captures and the fuller citation comparison.

#### 3. The NYT's timing claims need more context

Part of the New York Times case against Back is about timing. It says he went quiet on the main cryptography mailing list while Satoshi was active, and only started talking about Bitcoin six weeks after Satoshi disappeared. Both would be suspicious if true, so I checked them against the list archives. The archives complicate both points: Back kept posting while Satoshi was active, and his first Bitcoin comment on the later randombit list came soon after Bitcoin first appeared there.

- **The original Cryptography list, at metzdowd.com.** [Back's silence there began in November 2007](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/lists/metzdowd-adam-back-counts.txt), nine months before Satoshi's first known email. He posted there twice in March 2010. He also [posted 15 times](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/lists/randombit/back-randombit.json) on the later randombit Cryptography list between March 2010 and February 2011, while Satoshi was still active.
- **The later Cryptography list, at randombit.net.** The [archive search](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/nyt-review.md) found no Bitcoin mention there before 9 June 2011; Back's first was 12 June. This gives his silence on that particular list a less suspicious context. It says nothing about all his other communications. The original metzdowd list had discussed Bitcoin since [Satoshi's announcement in October 2008](https://www.metzdowd.com/pipermail/cryptography/2008-October/014810.html).
- **The 2015 "Satoshi" email.** The [archived headers](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/claims/2015-08-15-vistomail-raw.eml) show a vistomail server in the sending path. That does not authenticate the person controlling the account, so it cannot by itself establish that Satoshi was alive in 2015.
- **No machine-metadata analysis.** The NYT piece has no time-zone or machine-metadata analysis.

**Evidence and novelty.** The [NYT claim-by-claim review](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/nyt-review.md) identifies the list records and header evidence. I found no earlier critique making these list-timing comparisons. The vistomail observation had already been discussed by [Barely Sociable](https://www.youtube.com/watch?v=XfcvX0P1b5g).

#### 4. Satoshi's last emails may mark a change of routine

Satoshi rarely posted in the UTC morning or early afternoon in the earlier record. Some of his final emails appear to break that pattern, although their time zones are less certain.

- In the 2008–2010 baseline, [25 of 887 distinct, timed events](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/satoshi/check-2011-break.txt) fall in the quietest eight-hour window, 06:00–14:00 UTC. In January–April 2011, 4 of 11 timed items fall there under the research corpus's timezone assumptions.
- Three of those four are emails to Mike Hearn, whose [published copies](https://github.com/lextoumbourou/opus-satoshi-research/tree/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/satoshi/raw/private_emails/hearn) omit the timezone. The analysis assumes Zurich time, supported by his work location and a fit to earlier emails, but that fit is not an independent measurement.
- A simple [binomial calculation](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/scripts/check_2011_break.py) gives p ≈ 0.00018. It treats the events as independent draws from a stable routine, uses a quiet window selected from the earlier sample, and treats the inferred email times as fixed. Those assumptions, plus the small and unevenly preserved sample, make this an exploratory statistic rather than strong evidence of a location or identity.
- Gavin Andresen's 26 April email gives a separate morning-UTC clue. Its time is [reconstructed from his published reply](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/2011-break.md), rather than the Hearn display-zone assumption.
- The April dates fall around Easter. Holiday schedules could explain a change, but the timing doesn't distinguish the UK from nearby European countries.

**Evidence and novelty.** The [2011 analysis](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/2011-break.md) and [script](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/scripts/check_2011_break.py) show the assumptions and timezone sensitivity. I found no previous analysis of this particular shift. A change of routine remains a possibility, with limited evidence for a geographic interpretation.

#### 5. Adam Back's own rhythm doesn't match Satoshi's

If Back were Satoshi, you might expect the two to keep similar hours. So I collected 1,455 of Back's posts, from 1996 to 2021, and compared their timing with Satoshi's 2008–2011 record. These periods and venues differ, which limits the comparison. They don't match: Back often posts in the UK morning, which Satoshi almost never did. Some contemporaneous emails also show different timezone settings for Back and Satoshi.

- **Posting times (verified computation).** I looked at [1,455 of Back's posts](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/candidates/back-activity/README.md) across four venues: the Cryptography list (2010–15), Bitcointalk, bitcoin-dev, and the cypherpunks list (1996–98). [Between 28% and 49% of his posts](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/candidates/back-activity/rhythm-extended.txt) fall in Satoshi's quiet window, against 3% for Satoshi. Even in his UK years, 18.5% of his posts are between 9am and 1pm UK time. Satoshi's figure for those hours is 0.3%.
- **Different timezone settings (verified headers; inferred locations).** Back's [email records](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/timezone-evidence.md#relevance-to-named-candidates-inference) show UK time in the 1990s, US Eastern in his Montreal years, and Central European time in Malta from at least March 2010. On 1 December 2010 and 25 January 2011, Back's emails are stamped +0100 while Satoshi's are +0000.
- **Writing habits (verified, small sample).** I [tested](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/scripts/writing_habits_check.py) a claim from a [Hacker News comment](https://news.ycombinator.com/item?id=47697328) on 21 of Back's 2010–11 list posts. He writes bracketed full sentences with the full stop outside, uses "(ie" and "(eg" without dots, and uses "nor", at [about 5–11 per 10,000 words](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/data/candidates/back-activity/writing-habits.txt). In 90,000 words, Satoshi does the first once and the others never.
- None of this excludes a disciplined person running a separate persona on a separate machine. Like the case for him, it's circumstantial.

**Evidence and novelty.** The [four-venue comparison](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/back-rhythm-extended.md), [script](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/scripts/back_rhythm_extended.py) and [clock records](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/timezone-evidence.md) show the samples and exclusions. I found no prior publication of this combined comparison. The writing-habit claim came from a Hacker News commenter; this was a small follow-up check.

### What this means for the collective theory

Putting it all together, here's how each member of the proposed collective fares against the evidence. In short, nothing rules out a group sharing ideas or code. But the person actually running Satoshi day to day, writing the emails and building the software, used a fairly consistent working environment. That is compatible with one operator, but it does not establish how many people had access to that environment.

- **Back.** His links as an influence and as the first person Satoshi wrote to are verified. As the day-to-day Satoshi, though, he'd have needed a [separate Windows machine kept on UK time](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/timezone-evidence.md#relevance-to-named-candidates-inference) for over a year after he moved to Malta, plus a posting routine unlike his own. Separate machines and separate routines are plausible for someone maintaining a pseudonym, so this does not exclude him.
- **Finney.** He's a documented helper and tester (verified). His own machines were on [California time](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/release-forensics.md). The October 2008 draft PDF's [−07:00](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/timezone-evidence.md#summary-table) is Pacific daylight time, which fits him, but the 2009–10 mail and build machines don't. This would require separate timezone settings for the two identities, which is possible.
- **Sassaman.** He lived in Belgium, which is on Central European time, not UK time.
  - He spoke at Black Hat in Las Vegas on 28 July 2010. Nearby Satoshi email and build timestamps used UK summer time. This is a difference in settings, not a demonstrated simultaneous-activity alibi.
  - The inspected Sassaman software packages were made on Linux; Satoshi's Windows packages were made on Windows. That does not show either person used only one operating system.
  - The Belgian citation doesn't point to him.
  - These observations do not exclude him as an author or contributor.
- **Szabo.** Nothing I found tests him directly.
- **The keys argument.** Some argue that Sassaman held Satoshi's keys, which is why the coins never moved after he died. [Inactivity in coins attributed to Satoshi](https://bitslog.com/2019/04/16/the-return-of-the-deniers-and-the-revenge-of-patoshi/) began before Sassaman's death, so their remaining still afterwards is not specific evidence for him. This argument also depends on which coins are attributed to Satoshi.
- **The collective overall.** It can't be ruled out for ideas, review or code contributions. A consistently configured account or machine does not establish a single author. The evidence here cannot settle the number of contributors.

The [packaging comparison](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/packaging-habits.md) and [Black Hat timing check](https://github.com/lextoumbourou/opus-satoshi-research/blob/5c5528bb8f765847384e9edaf535fd2ac1d6888a/notes/sassaman-blackhat-2010.md) contain the sources behind those narrower observations.

### What would move the needle

None of this settles the question. A few things could, and most of them are already in someone's hands:

- **The original Satoshi emails with full headers**, held by Malmi, Andresen, Hearn and others. Mail servers often record the sending IP address in the Received headers. A trustworthy sending address could help test network location, although a relay, VPN or Tor exit would not establish the sender's physical location.
- **Who set up the London-time build box** that Gavin used in 2011.
- **Proper stylometry on the white paper itself** against each candidate's formal writing, with a method fixed in advance.

---

I guess the mystery will live on.
