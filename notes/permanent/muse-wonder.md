---
title: "Muse Wonder: Opus 5.5 quality model that supports audio"
slug: muse-wonder
date: 2026-10-13 06:34
modified: 2026-10-13 12:59
category: model
cover: /_media/muse-wonder-cover.jpg
cover_credits: Photo by <a href="https://unsplash.com/@samuelqyx?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Samuel Quek</a> on <a href="https://unsplash.com/photos/silhouette-of-a-person-looking-at-the-milky-way-cD4pt47I4JY?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
bluesky_post: https://bsky.app/profile/notesbylex.com/post/3mxpzy6snmy26
mastodon_post: https://fedi.notesbylex.com/@lex/117431356149010859
threads_post: https://www.threads.com/@lexisoninsta/post/DeayhOAlMSk
linkedin_post: https://www.linkedin.com/posts/lextoumbourou_shared-some-notes-on-metas-new-frontier-share-7515580697181360128-k4KN/
summary: Meta's new frontier model can listen to audio, and costs the same per token as Claude Opus 5.5 while using fewer tokens.
aliases:
- Muse Wonder
tags:
  - ModelRelease
  - Meta
notebook:
  timeout: 1200000
  outputLimit: 200
---

Yesterday (US time), Meta [released](https://research.meta.ai/blog/muse-wonder) two new frontier models, **Muse Wonder** and its bigger sibling **Muse Wonder Full** (*wonderful* - get it?).

Meta pitches them as "personal and professional agents", built around two features:

- **Active Memory**, which carries context about you across conversations (see [Memory](memory.md)).
- **Active Focus**, which sticks with a long task, works through blockers and checks in before important actions.

Although, I'm pretty sure these features aren't available through the API (yet?) [@metaModelAPIModels].

Only Muse Wonder is available in the API. Muse Wonder Full needs a Muse Code subscription (the Power Usage or Maximum Usage plan), so it has no per-token price [@metaBuildWithMuseWonder].

One cool thing is that these models support audio directly, alongside text, images, video and PDFs [@metaModelAPIModels]. I've always wondered when Claude models would support audio. I'm sure it's something they're looking at.

The model capability is very similar to [Claude Opus 5.5](claude-opus-5-5.md), so I use that as the comparison for this article.

## Specs Muse Wonder vs Opus 5.5

| Specification    | Muse Wonder                                    | Claude Opus 5.5                  |
| ---------------- | ---------------------------------------------- | -------------------------------- |
| API identifier   | `muse-wonder-1.0`                              | `claude-opus-5-5`                |
| Input / output   | Text, images, video, audio and PDF / text      | Text and images / text           |
| Context window   | 1,048,576 tokens                               | 1M tokens                        |
| Maximum output   | Not published                                  | 128K tokens                      |
| Knowledge cutoff | Not published                                  | June 2026                        |
| Reasoning        | Always on; `none` returns an error             | Adaptive thinking, always on     |
| Effort levels    | `minimal` to `max`, default not published      | `low` to `max`, default `medium` |
| Status           | Preview                                        | Generally available              |

As of 13 October 2026 [@metaModelAPIModels] [@metaModelAPIReasoning] [@anthropicModelsOverview].

## Pricing and Cache Settings

It's also priced exactly the same per token as Opus 5.5, down to the cache-read rate, so I compare the two throughout.

Prices in USD per million tokens, as of 13 October 2026.

| Usage | Muse Wonder | Claude Opus 5.5 |
|---|---:|---:|
| Input, uncached | $4.00 | $4.00 |
| Input, cache read | $0.20 | $0.20 |
| Cache write | No price listed | $5.00 (5 min), $8.00 (1 hour) |
| Output, including reasoning | $20.00 | $20.00 |
| Batch input / output | Not listed | $2.00 / $10.00 |
| Web search | $2.50 per 1,000 searches | $10 per 1,000 searches |

Neither charges a long-context premium, so a nearly full 1M-token window costs the same per token as a short prompt [@metaModelAPIPricing] [@anthropicPricing].

The caching works differently, though. Meta's caching is implicit. That is, it runs on every request, with no breakpoints to mark and no separate cache-write price listed. You can pass a `prompt_cache_key` to improve hit rates, or ask for 24-hour retention with `prompt_cache_retention`, but Meta describes the retention setting as "a hint, not a guarantee" [@metaModelAPIPromptCaching].

On Opus you opt in with `cache_control` and pay 1.25x (5 minutes) or 2x (1 hour) the input price to write the cache [@anthropicPricing].

## Tokenisers

I counted the same six files I used for [Claude Haiku 5.5](claude-haiku-5-5.md) with each provider's token-counting endpoint (Anthropic's `count_tokens` and Meta's `/v1/responses/input_tokens` [@metaModelAPITokenCounting]) on 13 October 2026:

| Sample | Opus 5.5 tokens | Muse Wonder tokens | Ratio |
|---|---:|---:|---:|
| Survey summary (prose) | 9,992 | 6,064 | 1.65 |
| AudioLM summary (prose) | 8,024 | 4,946 | 1.62 |
| Ansible tutorial (prose and commands) | 6,086 | 4,109 | 1.48 |
| [GPT 6 Sol and Luna](gpt-6-sol-and-luna.md) article (prose and code) | 13,847 | 8,990 | 1.54 |
| Python rendering script | 15,708 | 10,119 | 1.55 |
| Generated HTML game | 11,393 | 7,762 | 1.47 |

So Opus produces about 1.5 to 1.65 times as many tokens for the same text. At identical per-token prices, Muse Wonder is about 1.5 to 1.65 times cheaper for the same input, and output is likely similar. Muse Wonder's counts are within about 2% of GPT-6 Luna's from the Haiku article.

## Cost Per Task

Artificial Analysis gives a better sense of what you actually pay. Running its Intelligence Index, both at max effort [@artificialAnalysisMuseWonder] [@artificialAnalysisClaudeOpus55]:

| | Muse Wonder (max) | Claude Opus 5.5 (max) |
|---|---:|---:|
| Intelligence Index v4.3 | 52 | 58 |
| Output tokens used | 130M | 260M |
| Cost per index task | $4.52 | $5.98 |

So Opus scores higher, but uses twice the output tokens and costs about a third more per task. Artificial Analysis also measured Muse Wonder's time to first token at 19.27 seconds on Meta's API, so it's not fast to get going [@artificialAnalysisMuseWonder].

## Testing the audio

### Classifying sounds

I took 25 five-second clips from ESC-50, a sound classification dataset with 50 labels like dog, rain, siren and church bells [@piczakESCDatasetEnvironmental2015]. I sent each clip to the API with the full list of labels and asked it to pick one [@metaModelAPIVideoUnderstanding].

It got **20 of 25 right (80%)**. Guessing would get 2%, and human listeners in the original paper scored 81.3% [@piczakESCDatasetEnvironmental2015].

| Clip | Muse Wonder's answer |
|---|---|
| [Pig](https://freesound.org/s/208757/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-pig.mp3"></audio> | Pig ✓ |
| [Snoring](https://freesound.org/s/20545/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-snoring.mp3"></audio> | Dog ✗ |
| [Siren](https://freesound.org/s/31482/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-siren.mp3"></audio> | Siren ✓ |
| [Helicopter](https://freesound.org/s/172649/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-helicopter.mp3"></audio> | Rain ✗ |
| [Laughing](https://freesound.org/s/1791/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-laughing.mp3"></audio> | Laughing ✓ |

<details markdown="1">
<summary>All 25 clips</summary>

| Clip | Muse Wonder's answer |
|---|---|
| [Dog](https://freesound.org/s/100032/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-dog.mp3"></audio> | Dog ✓ |
| [Pig](https://freesound.org/s/208757/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-pig.mp3"></audio> | Pig ✓ |
| [Frog](https://freesound.org/s/15689/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-frog.mp3"></audio> | Frog ✓ |
| [Hen](https://freesound.org/s/18074/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-hen.mp3"></audio> | Hen ✓ |
| [Sheep](https://freesound.org/s/121951/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-sheep.mp3"></audio> | Sheep ✓ |
| [Rain](https://freesound.org/s/17367/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-rain.mp3"></audio> | Rain ✓ |
| [Crackling fire](https://freesound.org/s/17150/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-crackling-fire.mp3"></audio> | Crackling fire ✓ |
| [Chirping birds](https://freesound.org/s/100038/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-chirping-birds.mp3"></audio> | Chirping birds ✓ |
| [Wind](https://freesound.org/s/137296/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-wind.mp3"></audio> | Sea waves ✗ |
| [Toilet flush](https://freesound.org/s/20736/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-toilet-flush.mp3"></audio> | Toilet flush ✓ |
| [Crying baby](https://freesound.org/s/187207/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-crying-baby.mp3"></audio> | Crying baby ✓ |
| [Clapping](https://freesound.org/s/104089/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-clapping.mp3"></audio> | Clapping ✓ |
| [Coughing](https://freesound.org/s/19111/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-coughing.mp3"></audio> | Coughing ✓ |
| [Laughing](https://freesound.org/s/1791/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-laughing.mp3"></audio> | Laughing ✓ |
| [Snoring](https://freesound.org/s/20545/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-snoring.mp3"></audio> | Dog ✗ |
| [Door wood knock](https://freesound.org/s/101336/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-door-wood-knock.mp3"></audio> | Door wood knock ✓ |
| [Keyboard typing](https://freesound.org/s/137/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-keyboard-typing.mp3"></audio> | Dog ✗ |
| [Can opening](https://freesound.org/s/101404/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-can-opening.mp3"></audio> | Can opening ✓ |
| [Vacuum cleaner](https://freesound.org/s/100210/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-vacuum-cleaner.mp3"></audio> | Vacuum cleaner ✓ |
| [Clock tick](https://freesound.org/s/21934/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-clock-tick.mp3"></audio> | Clock tick ✓ |
| [Helicopter](https://freesound.org/s/172649/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-helicopter.mp3"></audio> | Rain ✗ |
| [Siren](https://freesound.org/s/31482/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-siren.mp3"></audio> | Siren ✓ |
| [Engine](https://freesound.org/s/18527/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-engine.mp3"></audio> | Engine ✓ |
| [Church bells](https://freesound.org/s/13571/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-church-bells.mp3"></audio> | Dog ✗ |
| [Fireworks](https://freesound.org/s/115545/)<br><audio controls preload="none" style="width:100%;min-width:150px" src="../_media/muse-wonder-esc50-fireworks.mp3"></audio> | Fireworks ✓ |

</details>


Clips from ESC-50 (CC BY-NC 3.0), originally from [Freesound](https://freesound.org).

Three of the misses were "dog", the first label in the list. When I asked it to describe those clips, it often said "no recording was attached", even though the audio was sent. So sometimes it just doesn't hear the clip, and it doesn't tell you.

The 25 clips cost about US$0.69, mostly in thinking ([raw results](../_media/muse-wonder-esc50-results.json)).

### Transcribing two speakers

I also asked it to transcribe 75 seconds of a NotebookLM Audio Overview about Muse Wonder, starting a new line at each change of speaker.

<audio controls preload="none" src="../_media/muse-wonder-transcription-clip.mp3"></audio>

It was basically perfect, with only one or two words off, and it even caught a quiet "Right, yeah" that WhisperX missed. Each run cost about US$0.05 ([raw results](../_media/muse-wonder-transcription-results.json)).

## Benchmarks

Meta's model card compares both Muse Wonder models against Claude Fable 5.1, GPT-6 Astra and Claude Opus 5.5, with every model at max reasoning effort [@metaMuseWonderModelCard]. A selection of rows from its scorecard:

| Benchmark                                        | Muse Wonder Full | Muse Wonder | Opus 5.5 | Fable 5.1 | GPT-6 Astra |
|---|---:|---:|---:|---:|---:|
| Personal Superintelligence Index (Meta internal) | **59.1** | 54.3 | 55.2 | 49.9 | 54.5 |
| Harvey LAB-AA | **13.1** | 10.0 | 4.2 | 6.4 | 8.6 |
| MemoryBench (Meta internal) | 85.5 | **85.6** | 81.5 | 79.0 | 81.4 |
| AutomationBench-AA | **45.4** | 39.1 | 41.2 | 32.1 | 41.6 |
| OSWorld V2.1, partial reward | 78.4 | 72.6 | **81.8** | 80.7 | 75.8 |
| [Terminal-Bench](terminal-bench.md) 4.0-AA | 57.1 | 52.0 | **59.6** | 52.0 | 59.1 |
| DeepSWE 1.1 | **79.0** | 78.8 | 74.2 | 67.4 | 74.1 |
| SWE-Marathon | **57.5** | 53.1 | 46.3 | 45.6 | 42.5 |
| ProgramBench | 59.0 | 50.0 | **70.5** | 55.5 | 52.5 |
| Artificial Analysis Intelligence Index | 56 | 52 | **58** | 53 | 53 |

Even in Meta's own table, Opus 5.5 wins on computer use, terminal coding and ProgramBench. Muse Wonder wins on legal work, memory and long-running software tasks, though the biggest margins belong to Muse Wonder Full, which isn't on the API. Some of these benchmarks are Meta's own, and not every row used the same [Coding Harness](coding-harness.md) [@metaMuseWonderModelCard].

Overall, Opus 5.5 still looks like the stronger model at the same price per token, but Muse Wonder may be cheaper per task, and it's worth trying for long-running agent work.

## Real example

As with [Claude Opus 5.5](claude-opus-5-5.md#real-example) and [GPT 6 Sol and Luna](gpt-6-sol-and-luna.md#real-example), I asked Muse Wonder at high effort to turn this article into a 3D gameshow with an N64 look, using the same prompt.

Meta's API is compatible with the [OpenAI Python SDK](https://github.com/openai/openai-python), so the call is the same as the GPT 6 Sol example with a different `base_url` [@metaModelAPIModels]. The [Notebook HTML helper](../_media/notesbylex_notebook_html-0.1.2-py3-none-any.whl) saves the game and token usage, then renders it:

```bash
pip install openai ./notesbylex_notebook_html-0.1.2-py3-none-any.whl
```

Set `MODEL_API_KEY` in the environment used to launch Obsidian. Running the cell makes a paid request; the saved game plays without an API key or another model call.

```python {id=muse-wonder-quiz}
import os
from openai import OpenAI
from notebook_html import read_article, render_openai_html

client = OpenAI(api_key=os.environ["MODEL_API_KEY"], base_url="https://api.meta.ai/v1")
article = read_article("muse-wonder.md", before="Real example")
with client.responses.stream(
    model="muse-wonder-1.0",
    max_output_tokens=128_000,
    reasoning={"effort": "high"},
    instructions="Return a complete, self-contained HTML document.",
    input=f"""
Turn this article into a 3D gameshow that looks like an N64 game.
Use around 8 questions that test understanding of the article, with a short
explanation after each answer.
Support phones and keyboards. Use inline CSS and JavaScript, no external assets,
network requests or browser storage, and no em dashes. Keep the code compact,
with one shared question renderer.

Article:
{article}
""",
) as stream:
    response = stream.get_final_response()
render_openai_html(response, "../_media/muse-wonder-quiz-high.html")
```
<!-- nb-output id="muse-wonder-quiz" hash="e96d73f0edda93e9" format="html" -->
<div class="nb-output">
<div class="nb-output-html"><iframe title="Generated HTML" width="100%" height="900" style="border:0;display:block" sandbox="allow-scripts" srcdoc="&lt;!doctype html&gt;
&lt;html lang=&quot;en&quot;&gt;
&lt;head&gt;
&lt;meta charset=&quot;utf-8&quot;&gt;
&lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,initial-scale=1&quot;&gt;
&lt;title&gt;Muse Wonder Showdown 64&lt;/title&gt;
&lt;style&gt;
*{box-sizing:border-box}body{margin:0;background:#08041f;color:#fff;font-family:Arial Black,Arial,sans-serif;overflow-x:hidden}
body:before{content:&quot;&quot;;position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent 0 3px,rgba(0,0,0,.18) 4px);pointer-events:none;z-index:9}
button{font:inherit;cursor:pointer}.wrap{max-width:980px;margin:auto;padding:12px}
.top{display:flex;justify-content:space-between;gap:8px;align-items:center}.logo{font-size:clamp(22px,5vw,46px);line-height:.9;color:#ffe600;text-shadow:4px 4px #e6007e,-3px -2px #00cfff;letter-spacing:-1px;transform:skew(-6deg)}
.logo small{display:block;font-size:.32em;color:#fff;text-shadow:2px 2px #333;letter-spacing:3px}
.hud{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}.pill{background:#22105e;border:3px solid #fff;box-shadow:4px 4px #000;padding:7px 10px;font-size:13px}
.pill b{color:#ffe600}
.scene{height:300px;perspective:700px;overflow:hidden;position:relative;background:linear-gradient(#13075c 45%,#5b1685 70%,#15052e);border:5px solid #fff;box-shadow:8px 8px #000;margin-top:12px}
.scene:before{content:&quot;MUSE WONDER&quot;;position:absolute;left:50%;top:24px;transform:translateX(-50%);font-size:clamp(24px,6vw,56px);color:#ffea00;text-shadow:4px 4px #f0f,-4px 3px #0ff;white-space:nowrap;opacity:.9}
.star{position:absolute;width:18px;height:18px;background:#ff0;clip-path:polygon(50% 0,65% 35%,100% 50%,65% 65%,50% 100%,35% 65%,0 50%,35% 35%);animation:spin 3s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.world{position:absolute;inset:0;transform-style:preserve-3d;animation:sway 6s ease-in-out infinite alternate}
@keyframes sway{from{transform:rotateY(-5deg)}to{transform:rotateY(5deg)}}
.floor{position:absolute;width:760px;height:360px;left:50%;bottom:-90px;background:repeating-linear-gradient(0deg,#00b7eb 0 4px,transparent 4px 45px),repeating-linear-gradient(90deg,#e600ff 0 4px,transparent 4px 55px),#18094d;transform:translateX(-50%) rotateX(68deg);transform-origin:bottom}
.host{position:absolute;left:50%;top:92px;transform:translateX(-50%);text-align:center;animation:bob 1.3s ease-in-out infinite alternate;z-index:2}
@keyframes bob{to{transform:translateX(-50%) translateY(-8px)}}
.head{width:92px;height:82px;margin:auto;background:linear-gradient(135deg,#ffd166 55%,#f49f0a);border:5px solid #111;display:grid;place-items:center;font-size:27px;color:#111;box-shadow:inset -12px -12px rgba(0,0,0,.18)}
.body{width:110px;height:58px;background:linear-gradient(90deg,#e60023 50%,#ff5d73 50%);border:5px solid #111;margin:-3px auto 0;clip-path:polygon(15% 0,85% 0,100% 100%,0 100%)}
.podiums{position:absolute;bottom:18px;left:0;right:0;display:flex;justify-content:center;gap:clamp(18px,8vw,90px);z-index:2}
.pod{text-align:center;filter:drop-shadow(7px 8px #000)}
.pod i{display:block;width:66px;height:52px;margin:auto;background:#00d5ff;border:4px solid #111;transform:skewY(-4deg);font-style:normal;padding-top:12px;color:#001}
.pod span{display:block;width:82px;height:45px;background:#4720a8;border:4px solid #111;margin-top:-3px}
.pod.you i{background:#ffe600}.pod.you{transform:scale(1.18)}
.say{position:absolute;right:4%;top:78px;max-width:220px;background:#fff;color:#111;border:4px solid #111;padding:9px;font-family:Arial;font-weight:bold;font-size:14px;box-shadow:5px 5px #000;z-index:3}
.say:after{content:&quot;&quot;;position:absolute;left:-14px;top:35px;border:7px solid transparent;border-right-color:#111}
.panel{background:#17094b;border:5px solid #fff;box-shadow:8px 8px #000;padding:16px;margin-top:14px;position:relative}
.timer{height:16px;background:#111;border:3px solid #fff;margin-bottom:12px}.timer i{display:block;height:100%;width:100%;background:linear-gradient(90deg,#00ff66,#ffe600,#ff2222);transition:width .2s linear}
.q{font-size:clamp(20px,3.6vw,31px);line-height:1.15;min-height:72px}
.answers{display:grid;grid-template-columns:1fr 1fr;gap:11px;margin-top:14px}
.ans{text-align:left;color:#fff;border:4px solid #111;box-shadow:5px 5px #000;padding:15px;min-height:70px;font-size:clamp(14px,2.5vw,18px);line-height:1.25}
.ans:active{transform:translate(3px,3px);box-shadow:2px 2px #000}.ans:focus-visible,.big:focus-visible{outline:4px solid #fff;outline-offset:3px}
.ans:nth-child(1){background:#0057ff}.ans:nth-child(2){background:#009b32}.ans:nth-child(3){background:#d9a400;color:#111}.ans:nth-child(4){background:#c4007a}
.ans .k{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:50%;background:#fff;color:#111;border:3px solid #111;margin-right:8px}
.ans.good{background:#00c853!important;color:#fff}.ans.bad{background:#d50000!important;color:#fff}.ans:disabled{opacity:.9;cursor:default}
.explain{display:none;background:#fff8b5;color:#111;border:4px solid #111;padding:13px;margin-top:14px;font-family:Arial;font-weight:bold;line-height:1.45;box-shadow:5px 5px #000}
.explain.show{display:block}.explain b{font-family:Arial Black}
.row{display:flex;gap:10px;justify-content:space-between;align-items:center;margin-top:14px;flex-wrap:wrap}
.big{background:#ffe600;color:#111;border:4px solid #111;box-shadow:5px 5px #000;padding:13px 20px;font-size:18px}
.big:active{transform:translate(3px,3px);box-shadow:2px 2px #000}.hint{font-family:Arial;font-size:13px;font-weight:bold;color:#cde}
.overlay{position:absolute;inset:0;background:rgba(5,0,30,.82);display:grid;place-items:center;text-align:center;padding:20px;z-index:5}
.overlay h1{font-size:clamp(34px,8vw,72px);line-height:.9;color:#ffe600;text-shadow:5px 5px #e6007e;margin:0}
.blink{animation:blink 1s steps(2) infinite;color:#0ff}@keyframes blink{50%{opacity:0}}
.hide{display:none!important}.burst{position:absolute;inset:0;pointer-events:none;overflow:hidden;z-index:4}
.bit{position:absolute;width:16px;height:16px;animation:fall 1s ease-in forwards}@keyframes fall{to{transform:translateY(260px) rotate(540deg);opacity:0}}
@media(max-width:620px){.answers{grid-template-columns:1fr}.scene{height:260px}.say{display:none}.podiums{gap:28px}.q{min-height:0}.hud{font-size:11px}}
&lt;/style&gt;
&lt;/head&gt;
&lt;body&gt;
&lt;div class=&quot;wrap&quot;&gt;
&lt;header class=&quot;top&quot;&gt;
&lt;div class=&quot;logo&quot;&gt;SHOWDOWN 64&lt;small&gt;MUSE WONDER EDITION&lt;/small&gt;&lt;/div&gt;
&lt;div class=&quot;hud&quot;&gt;&lt;div class=&quot;pill&quot;&gt;SCORE &lt;b id=&quot;score&quot;&gt;0&lt;/b&gt;&lt;/div&gt;&lt;div class=&quot;pill&quot;&gt;ROUND &lt;b id=&quot;round&quot;&gt;-/8&lt;/b&gt;&lt;/div&gt;&lt;div class=&quot;pill&quot;&gt;STREAK &lt;b id=&quot;streak&quot;&gt;0&lt;/b&gt;&lt;/div&gt;&lt;/div&gt;
&lt;/header&gt;

&lt;section class=&quot;scene&quot; aria-label=&quot;3D game show stage&quot;&gt;
&lt;i class=&quot;star&quot; style=&quot;left:8%;top:16%&quot;&gt;&lt;/i&gt;&lt;i class=&quot;star&quot; style=&quot;right:10%;top:20%;animation-delay:.7s&quot;&gt;&lt;/i&gt;&lt;i class=&quot;star&quot; style=&quot;left:20%;top:48%&quot;&gt;&lt;/i&gt;
&lt;div class=&quot;world&quot;&gt;&lt;div class=&quot;floor&quot;&gt;&lt;/div&gt;
&lt;div class=&quot;host&quot;&gt;&lt;div class=&quot;head&quot;&gt;^_^&lt;/div&gt;&lt;div class=&quot;body&quot;&gt;&lt;/div&gt;&lt;/div&gt;
&lt;div class=&quot;podiums&quot;&gt;&lt;div class=&quot;pod&quot;&gt;&lt;i&gt;CPU&lt;/i&gt;&lt;span&gt;&lt;/span&gt;&lt;/div&gt;&lt;div class=&quot;pod you&quot;&gt;&lt;i&gt;YOU&lt;/i&gt;&lt;span&gt;&lt;/span&gt;&lt;/div&gt;&lt;div class=&quot;pod&quot;&gt;&lt;i&gt;CPU&lt;/i&gt;&lt;span&gt;&lt;/span&gt;&lt;/div&gt;&lt;/div&gt;
&lt;/div&gt;
&lt;div class=&quot;say&quot; id=&quot;say&quot;&gt;Welcome to the show. Pick the right answer, player!&lt;/div&gt;
&lt;div class=&quot;burst&quot; id=&quot;burst&quot;&gt;&lt;/div&gt;

&lt;div class=&quot;overlay&quot; id=&quot;startBox&quot;&gt;&lt;div&gt;&lt;h1&gt;MUSE&lt;br&gt;WONDER&lt;br&gt;SHOWDOWN 64&lt;/h1&gt;&lt;p&gt;8 questions. One champion. Low polygons, high stakes.&lt;/p&gt;&lt;button class=&quot;big&quot; id=&quot;startBtn&quot;&gt;▶ PRESS START&lt;/button&gt;&lt;p class=&quot;blink&quot;&gt;INSERT COIN - NO COIN NEEDED&lt;/p&gt;&lt;/div&gt;&lt;/div&gt;
&lt;div class=&quot;overlay hide&quot; id=&quot;endBox&quot;&gt;&lt;div&gt;&lt;h1 id=&quot;endTitle&quot;&gt;FINISH!&lt;/h1&gt;&lt;p id=&quot;endText&quot;&gt;&lt;/p&gt;&lt;button class=&quot;big&quot; id=&quot;againBtn&quot;&gt;↻ PLAY AGAIN&lt;/button&gt;&lt;/div&gt;&lt;/div&gt;
&lt;/section&gt;

&lt;main class=&quot;panel&quot;&gt;
&lt;div class=&quot;timer&quot;&gt;&lt;i id=&quot;timebar&quot;&gt;&lt;/i&gt;&lt;/div&gt;
&lt;div class=&quot;q&quot; id=&quot;question&quot;&gt;Press START to play.&lt;/div&gt;
&lt;div class=&quot;answers&quot; id=&quot;answers&quot;&gt;&lt;/div&gt;
&lt;div class=&quot;explain&quot; id=&quot;explain&quot;&gt;&lt;/div&gt;
&lt;div class=&quot;row&quot;&gt;&lt;span class=&quot;hint&quot;&gt;Phone: tap an answer. Keyboard: 1 to 4 or A to D, then Enter for Next.&lt;/span&gt;&lt;button class=&quot;big hide&quot; id=&quot;nextBtn&quot;&gt;NEXT ▶&lt;/button&gt;&lt;/div&gt;
&lt;/main&gt;
&lt;p class=&quot;hint&quot; style=&quot;text-align:center&quot;&gt;Built only from the article. No saves, no network, just glorious 64 bit judgement.&lt;/p&gt;
&lt;/div&gt;

&lt;script&gt;
const Q=[
{q:&quot;Which model can you actually call through the API?&quot;,a:[&quot;Both Muse Wonder models&quot;,&quot;Only Muse Wonder&quot;,&quot;Only Muse Wonder Full&quot;,&quot;Neither model&quot;],c:1,e:&quot;Only Muse Wonder is on the API. Muse Wonder Full is limited to Muse Code subscriptions on the Power Usage and Maximum Usage plans, with no per token price.&quot;},
{q:&quot;What do Active Memory and Active Focus mean in Meta&#x27;s pitch?&quot;,a:[&quot;Faster images and shorter prompts&quot;,&quot;Memory carries context across chats, Focus sticks with long tasks&quot;,&quot;Both names describe cheaper caching&quot;,&quot;Memory deletes context, Focus avoids check ins&quot;],c:1,e:&quot;Active Memory means carrying context about you across conversations. Active Focus means staying with a long task, working through blockers and checking in before important actions.&quot;},
{q:&quot;Per token, how does Muse Wonder compare with Claude Opus 5.5?&quot;,a:[&quot;Wonder costs exactly the same, including cache reads&quot;,&quot;Wonder is always half the price&quot;,&quot;Opus is cheaper on every line&quot;,&quot;Only their output prices match&quot;],c:0,e:&quot;The listed rates match exactly: input is $4, cache read is $0.20 and output is $20 per million tokens. That direct match is why the article compares them head to head.&quot;},
{q:&quot;Which input advantage does Muse Wonder list over Opus 5.5?&quot;,a:[&quot;It only accepts text&quot;,&quot;It accepts text, images, video, audio and PDF&quot;,&quot;Opus accepts video but Wonder does not&quot;,&quot;Both accept exactly the same inputs&quot;],c:1,e:&quot;Muse Wonder lists text, images, video, audio and PDF as input. Opus 5.5 lists text and images. Meta still warns users to check audio support on the surface they are using.&quot;},
{q:&quot;How is Meta&#x27;s caching different from Opus caching?&quot;,a:[&quot;Meta requires cache breakpoints&quot;,&quot;Opus caching is always automatic and free&quot;,&quot;Meta caching is implicit, with no separate cache write price listed&quot;,&quot;Both use the same cache control setting&quot;],c:2,e:&quot;Meta caching runs implicitly on every request, with no breakpoints to mark and no separate cache write price listed. On Opus, you opt in with cache_control and pay a cache write price.&quot;},
{q:&quot;Why can Muse Wonder be cheaper per task despite a lower Intelligence Index score?&quot;,a:[&quot;It uses half the output tokens and costs $4.52 versus $5.98 per task&quot;,&quot;It has a lower price per token&quot;,&quot;It skips reasoning entirely&quot;,&quot;Opus uses fewer tokens on every task&quot;],c:0,e:&quot;Artificial Analysis scored Wonder 52 and Opus 58. But Wonder used 130M output tokens versus 260M for Opus, so Wonder cost $4.52 per index task compared with $5.98.&quot;},
{q:&quot;Where does Muse Wonder beat Opus 5.5 in Meta&#x27;s scorecard?&quot;,a:[&quot;OSWorld, ProgramBench and Terminal-Bench&quot;,&quot;Legal work, memory and long horizon software engineering&quot;,&quot;Only the Artificial Analysis index&quot;,&quot;It wins every benchmark listed&quot;],c:1,e:&quot;Wonder&#x27;s wins include Harvey LAB legal work, MemoryBench and long horizon software work such as DeepSWE and SWE-Marathon. Opus still wins OSWorld, Terminal-Bench, ProgramBench and the Artificial Analysis index.&quot;},
{q:&quot;Why should some benchmark margins be treated with caution?&quot;,a:[&quot;All scores came from independent common tests&quot;,&quot;Several caveats apply, including Meta internal tests and mixed harnesses&quot;,&quot;The model card lists no sources at all&quot;,&quot;Only Opus scores were estimated&quot;],c:1,e:&quot;The Personal Superintelligence Index and MemoryBench are Meta internal evaluations. The SWE-Marathon row also mixes harnesses for some rivals, and competitor runs by Meta are described as best effort.&quot;}
];
let i=0,score=0,streak=0,locked=true,timer=null,left=20;
const $=id=&gt;document.getElementById(id),answers=$(&quot;answers&quot;),question=$(&quot;question&quot;),explain=$(&quot;explain&quot;),next=$(&quot;nextBtn&quot;);
function tone(f,d=.12){try{let c=new(window.AudioContext||window.webkitAudioContext)(),o=c.createOscillator(),g=c.createGain();o.frequency.value=f;o.type=&quot;square&quot;;g.gain.value=.06;o.connect(g);g.connect(c.destination);o.start();o.stop(c.currentTime+d)}catch(e){}}
function confetti(){let b=$(&quot;burst&quot;);b.innerHTML=&quot;&quot;;for(let n=0;n&lt;28;n++){let s=document.createElement(&quot;i&quot;);s.className=&quot;bit&quot;;s.style.left=Math.random()*100+&quot;%&quot;;s.style.top=&quot;-20px&quot;;s.style.background=[&quot;#ff0&quot;,&quot;#0ff&quot;,&quot;#f0f&quot;,&quot;#0f6&quot;][n%4];s.style.clipPath=n%2?&quot;polygon(50% 0,100% 100%,0 100%)&quot;:&quot;none&quot;;b.appendChild(s)}setTimeout(()=&gt;b.innerHTML=&quot;&quot;,1100)}
function startTimer(){clearInterval(timer);left=20;$(&quot;timebar&quot;).style.width=&quot;100%&quot;;timer=setInterval(()=&gt;{left-=.2;$(&quot;timebar&quot;).style.width=Math.max(0,left/20*100)+&quot;%&quot;;if(left&lt;=0){clearInterval(timer);pick(-1)}},200)}
function renderQ(){
 locked=false;explain.className=&quot;explain&quot;;next.classList.add(&quot;hide&quot;);
 $(&quot;round&quot;).textContent=(i+1)+&quot;/8&quot;;$(&quot;score&quot;).textContent=score;$(&quot;streak&quot;).textContent=streak;
 question.textContent=&quot;Q&quot;+(i+1)+&quot;. &quot;+Q[i].q;$(&quot;say&quot;).textContent=&quot;Question &quot;+(i+1)+&quot; is live. Choose wisely!&quot;;
 answers.innerHTML=&quot;&quot;;
 Q[i].a.forEach((t,n)=&gt;{let b=document.createElement(&quot;button&quot;);b.className=&quot;ans&quot;;b.innerHTML=&#x27;&lt;span class=&quot;k&quot;&gt;&#x27;+&quot;ABCD&quot;[n]+&quot;&lt;/span&gt;&quot;+t;b.onclick=()=&gt;pick(n);answers.appendChild(b)});
 startTimer();answers.children[0].focus({preventScroll:true});
}
function pick(n){
 if(locked)return;locked=true;clearInterval(timer);
 let ok=n===Q[i].c,btns=[...answers.children];
 btns.forEach((b,x)=&gt;{b.disabled=true;if(x===Q[i].c)b.classList.add(&quot;good&quot;);else if(x===n)b.classList.add(&quot;bad&quot;)});
 if(ok){let pts=100+Math.ceil(left)*5;score+=pts;streak++;tone(880);setTimeout(()=&gt;tone(1320),110);confetti();$(&quot;say&quot;).textContent=&quot;Correct! The crowd goes mildly polygonal!&quot;}else{streak=0;tone(160,.25);$(&quot;say&quot;).textContent=n&lt;0?&quot;Time up! The correct answer lights up green.&quot;:&quot;Not quite. The green answer is the winner.&quot;}
 $(&quot;score&quot;).textContent=score;$(&quot;streak&quot;).textContent=streak;
 explain.innerHTML=&quot;&lt;b&gt;&quot;+(ok?&quot;CORRECT. &quot;:&quot;ANSWER: &quot;+&quot;ABCD&quot;[Q[i].c]+&quot;. &quot;)+&quot;&lt;/b&gt;&quot;+Q[i].e;explain.classList.add(&quot;show&quot;);
 next.textContent=i===Q.length-1?&quot;SEE RESULTS ▶&quot;:&quot;NEXT ▶&quot;;next.classList.remove(&quot;hide&quot;);next.focus({preventScroll:true});
}
function nextQ(){i++;i&lt;Q.length?renderQ():finish()}
function finish(){
 $(&quot;round&quot;).textContent=&quot;8/8&quot;;question.textContent=&quot;Game complete. Final score: &quot;+score;
 answers.innerHTML=&quot;&quot;;explain.className=&quot;explain&quot;;next.classList.add(&quot;hide&quot;);$(&quot;say&quot;).textContent=&quot;That is the show. Thanks for playing!&quot;;
 $(&quot;endTitle&quot;).textContent=score&gt;=900?&quot;🏆 GRAND CHAMPION&quot;:score&gt;=600?&quot;⭐ STAR PLAYER&quot;:&quot;GAME OVER&quot;;
 $(&quot;endText&quot;).textContent=&quot;You scored &quot;+score+&quot; points. Article verdict: Opus 5.5 looks stronger overall at the same token price, but Muse Wonder may cost less per task and suits long running agent work where memory matters.&quot;;
 $(&quot;endBox&quot;).classList.remove(&quot;hide&quot;);tone(660);setTimeout(()=&gt;tone(990),130);
}
function start(){i=0;score=0;streak=0;$(&quot;startBox&quot;).classList.add(&quot;hide&quot;);$(&quot;endBox&quot;).classList.add(&quot;hide&quot;);renderQ();tone(523);setTimeout(()=&gt;tone(784),110)}
$(&quot;startBtn&quot;).onclick=start;$(&quot;againBtn&quot;).onclick=start;next.onclick=nextQ;
document.addEventListener(&quot;keydown&quot;,e=&gt;{
 let k=e.key.toLowerCase();
 if(!$(&quot;startBox&quot;).classList.contains(&quot;hide&quot;)&amp;&amp;(k===&quot;enter&quot;||k===&quot; &quot;)){e.preventDefault();start();return}
 if(!$(&quot;endBox&quot;).classList.contains(&quot;hide&quot;)&amp;&amp;(k===&quot;enter&quot;||k===&quot; &quot;)){e.preventDefault();start();return}
 if(!locked&amp;&amp;[&quot;1&quot;,&quot;2&quot;,&quot;3&quot;,&quot;4&quot;,&quot;a&quot;,&quot;b&quot;,&quot;c&quot;,&quot;d&quot;].includes(k))pick(&quot;1234abcd&quot;.indexOf(k)%4);
 else if(locked&amp;&amp;!next.classList.contains(&quot;hide&quot;)&amp;&amp;(k===&quot;enter&quot;||k===&quot; &quot;)){e.preventDefault();nextQ()}
});
&lt;/script&gt;
&lt;/body&gt;
&lt;/html&gt;"></iframe></div>
</div>
<!-- /nb-output -->

This game was generated on **13 October 2026** using **`muse-wonder-1.0` at high effort**. It cost about **US$0.12**, calculated from the [saved token usage](../_media/muse-wonder-quiz-high.json) and Meta's standard API prices [@metaModelAPIPricing]:

| Usage | Tokens | Estimated cost (USD) |
|---|---:|---:|
| Input, uncached | 1,858 | $0.0074 |
| Output, including reasoning | 5,835 | $0.1167 |
| **Total** | **7,693** | **$0.1241** |

There were no cache reads. Of the output tokens, **1,296 were reasoning tokens** and **4,539 were the HTML response**. Reasoning is billed at the output rate [@metaModelAPIReasoning].

For comparison, the [Claude Opus 5.5](claude-opus-5-5.md#real-example) used 49,136 tokens and cost about **US$0.96**, so this run was about **87% cheaper** at the same per-token prices. Opus spent 36,150 tokens thinking; Muse Wonder spent 1,296. The [GPT 6 Sol and Luna](gpt-6-sol-and-luna.md#real-example) was cheaper again at about US$0.09. Each run used the same prompt at high effort, but with its own article as input.

[Play the game in its own page](../_media/muse-wonder-quiz-high.html).
