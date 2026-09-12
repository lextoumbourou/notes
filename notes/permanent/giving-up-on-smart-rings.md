---
category: essay
title: Giving Up on Smart Rings
date: 2026-09-13 00:00
modified: 2026-09-13 08:31
summary: The downsides to wearing a fitness-tracking device on your finger
cover: /_media/the-downsides-to-fitness-tracking-rings.png
bluesky_post: https://bsky.app/profile/notesbylex.com/post/3mve3nbax6x23
mastodon_post: https://fedi.notesbylex.com/@lex/117260336057642738
threads_post: https://www.threads.com/@lexisoninsta/post/DdNBLeGgT5F
tags:
  - Fitness
  - Wearables
---

After about a year of wearing a smart ring, I've decided to throw in the towel.

I did really like my Oura Ring. Tracking my sleep and steps, amongst other things, has been really helpful for my health journey. [Symptom Radar](https://support.ouraring.com/hc/en-us/articles/35593651188115-Symptom-Radar) was also typically pretty accurate and usually let me know I was about to get a cold even before I noticed the symptoms.

But a smart **ring** has a few major downsides that a wrist-based device doesn't, and they weren't obvious to me when I first bought mine.

## 1. Your fingers change size and rings are not adjustable

Turns out your fingers store quite a bit of fat, and if the fitness-tracking device helps you successfully lose weight, the ring might no longer fit. I initially bought an Oura Ring in size 13 when I was around 94 kg. Then I lost about 14 kg and it became really loose. After picking up a new sizing kit for a replacement device, I learned I was down to a size 11.

Your fingers also change size with temperature. Cold makes your fingers smaller, whilst warmer temperatures do the opposite. Oura's [product guidance](https://support.ouraring.com/hc/en-us/articles/43395388251283-Product-Safety-Use) says that finger size also fluctuates with food and drink, exercise and altitude.

## 2. You have to take them off to lift weights

Many exercises, from pull-ups to the bench press, place the ring between your hand and a metal bar. Oura recommends [removing the ring during activities that involve friction, including weightlifting](https://support.ouraring.com/hc/en-us/articles/43395388251283-Product-Safety-Use), to avoid scratches. For me, tracking workouts was one of the primary reasons that I bought it in the first place.

## 3. They get in the way when you wash and generally use your hands

Rings are constantly exposed while washing your hands, showering, cleaning and gripping things. Even when the ring is water-resistant, water and soap can become trapped underneath it, so removing, drying and replacing it becomes another small source of daily friction - something I repeatedly forget to do, which might explain how I ended up with 2 broken devices which I explain below. The outer surface is also easy to scratch. Oura's care instructions acknowledge the issue of both [trapped moisture and scratching](https://support.ouraring.com/hc/en-us/articles/43395388251283-Product-Safety-Use).

---

I must admit, though, the catalyst for these realisations about smart rings was that both my original ring and its replacement stopped working. Bluetooth would connect, but neither ring would track anything, as if I wasn't wearing it.

The first ring stopped working around the same time it stopped fitting me, and a newer model had come out anyway, so I begrudgingly replaced it.

But the replacement stopped working just two weeks after I got it. Thankfully, Oura were willing to offer a refund for the second ring. I think I was just unlucky, and I'm really not trying to throw the company under the bus, but it made me take stock of wearing a smart ring and realise these downsides.

As for what's next, I've decided to try a fitness band: Google Fitbit Air. I really have not historically been a fan of wearing watches and stuff on my wrist, but damn, do I now realise it's a convenient place to put a tracking device (which my fitness-obsessed wife had been trying to convince me of since I became ring-pilled). Anyway, I'm nearly 40. Maybe that's the right age to get into wrist wear. Does my watch era await?

As for my [OpenClaw Setup](6-months-of-openclaw.md), it now reads from the [Google Health API](https://developers.google.com/health/data-types) instead of Oura's API. It was pretty trivial to have Codex cut over, except, sadly, Google Health doesn't expose Fitbit's Sleep Score or a readiness score yet (please fix, Google!).
