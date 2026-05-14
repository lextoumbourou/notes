---
title: "LLMs Corrupt Your Documents When You Delegate"
date: 2026-05-09 00:00
modified: 2026-05-09 00:00
category: reference/papers
cover: /_media/llms-corrupt-cover.png
hide_cover_in_article: true
summary: "A large-scale study on long-horizon document tasks."
youtube_video: https://youtu.be/eEAXbFL9BZM?si=qn2Fds8BASZR0T_7
bluesky_post: https://bsky.app/profile/notesbylex.com/post/3mlf6gyxguc2r
mastodon_post: https://fedi.notesbylex.com/@lex/116542213243149508
linkedin_post: https://www.linkedin.com/feed/update/urn:li:activity:7458690168427003904
tags:
- AgenticReasoning
- LimitationsofLLMs
---

*My notes on [LLMs Corrupt Your Documents When You Delegate](https://arxiv.org/pdf/2604.15597) by Philippe Laban, Tobias Schnabel and Jennifer Neville from Microsoft Research.*

An interesting paper from researchers at Microsoft.

They introduce a benchmark called [DELEGATE-52](../../permanent/delegate-52.md) that tests whether LLMs can safely carry out, what they call, "long-delegated workflows" for document editing across 52 domains. Every set of instructions in the benchmark is lossless and reversible, allowing the authors to measure how much each task degrades the file's information over multiple interactions.

They found that even the strongest frontier models, including Gemini 3.1 Pro, Claude 4.6 Opus, and GPT 5.4 (the paper was released before their successors), corrupted an average of about 25% of document content after 20 interactions. Across all tested models, average degradation was about 50% [@labanLLMsCorruptYour2026].

Python was the main exception. It was the only domain in which most models met the paper’s “delegation-ready” threshold, with 17 of 19 models scoring at least 98% after 20 interactions.

![Three examples of document degradation over 20 interactions: a Linux Kernel Architecture graph diagram losing nodes and edges, a 12-Shaft Twill Diamond textile pattern becoming corrupted, and an ActionBoy Palm Tree 3D object losing geometry. Each shows progressive corruption from interaction 4 to 20.](../../_media/llms-corrupt-figure-1-examples.png)

*Figure 1 from [@labanLLMsCorruptYour2026] shows examples of document degradation across different domains. The benchmark itself is text-only; the visual renderings are illustrative.*

Surprisingly, the degradation didn't happen gradually over instructions, but models would typically fail catastrophically after a certain number of steps. Stronger frontier models would fare better only by delaying the step at which the degradation occurs.

They also found that tool use did not prevent degradation. The tested models performed worse with tools, averaging an additional 6% of degradation.

## Measuring Document Corruption

To measure document corruption, they introduce a domain-specific [Document Similarity Measure](../../permanent/document-similarity-measure.md) that parses documents into components. For a recipe, that means ingredients (name, quantity, unit), steps, and tips; for Python code, it means functions, classes, and imports. This lets them compare two parsed documents based on their actual content, rather than just raw text. Typical document similarity measures might overlook seemingly small changes, such as `200g` to `800g` of butter, which can be really bad in a recipe, whereas a surface-level rewrite that preserves the underlying structure doesn't need to be heavily penalised.

![Pipeline diagram showing how a raw recipe text file is parsed into structured ingredients, steps and tips, then scored for semantic equivalence against a reference using a weighted formula: 0.4 times ingredient score plus 0.4 times step score plus 0.2 times tip score.](../../_media/llms-corrupt-figure-5-document-parsing-similarity-score.png)

*Figure 5 from [@labanLLMsCorruptYour2026] - the domain-specific parsing pipeline, with a concrete recipe example showing how ingredients, steps and tips are extracted and compared*

The approach of creating reversible transforms was inspired by [Backtranslation](../../permanent/backtranslation.md), a machine translation technique in which text is translated into another language and then back, allowing the result to be compared with the original. DELEGATE-52 adapts that idea to document editing: apply a forward edit, apply the inverse edit, and compare the reconstructed document to the original. Imagine splitting a CSV into separate files by expense category, then merging them back together. Or converting all amounts in an accounting ledger to euros, then converting back.

They use a round-trip relay simulation method in which every task is assumed to be reversible, defined by a forward instruction and its inverse.

<div class="d52-widget">
<style>
.d52-widget{--d52-ink:#1a1208;--d52-paper:#f5f0e8;--d52-paper-warm:#ede8dc;--d52-rule:#c8bfa8;--d52-amber:#d4700a;--d52-amber-light:#f5d49a;--d52-amber-pale:#fdf3dc;--d52-green:#2d6a4f;--d52-green-light:#b7e4c7;--d52-red:#9b2335;--d52-red-light:#f4b8c1;--d52-blue-mid:#2563a8;--d52-muted:#7a6e5f;font-family:Georgia,'Times New Roman',serif}
.d52-widget *{box-sizing:border-box}
.d52-rtp{display:flex;align-items:center;gap:0;overflow-x:auto;padding:1.5rem 0 0.5rem;flex-wrap:nowrap}
.d52-node{flex-shrink:0;display:flex;flex-direction:column;align-items:center;gap:.4rem}
.d52-doc{background:#fff;border:1.5px solid var(--d52-ink);border-radius:3px;padding:.65rem .85rem;font-family:'Courier New',monospace;font-size:.7rem;line-height:1.65;min-width:110px;max-width:128px;box-shadow:3px 3px 0 var(--d52-ink)}
.d52-doc.corrupt{border-color:var(--d52-amber);box-shadow:3px 3px 0 var(--d52-amber)}
.d52-doc-title{font-weight:700;font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;border-bottom:1px solid #ccc;padding-bottom:.2rem;margin-bottom:.3rem;font-family:'Helvetica Neue',sans-serif}
.d52-dline{color:#444;margin-bottom:.05rem}
.d52-dline.cx{color:var(--d52-amber);font-weight:700}
.d52-arr{flex-shrink:0;display:flex;flex-direction:column;align-items:center;padding:0 .35rem;min-width:80px}
.d52-llm{background:var(--d52-ink);color:var(--d52-paper);border-radius:4px;padding:.4rem .55rem;font-family:'Helvetica Neue',sans-serif;font-size:.62rem;text-align:center;line-height:1.4;min-width:80px}
.d52-badge{display:block;background:var(--d52-amber);color:var(--d52-ink);font-size:.55rem;font-weight:700;letter-spacing:.05em;padding:.1rem .3rem;border-radius:2px;margin-top:.25rem;text-transform:uppercase}
.d52-aline{width:2px;height:14px;background:var(--d52-ink)}
.d52-achev{width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:7px solid var(--d52-ink)}
.d52-node-lbl{font-family:'Helvetica Neue',sans-serif;font-size:.65rem;color:var(--d52-muted);text-align:center;max-width:128px}
.d52-score{display:inline-flex;align-items:center;gap:.3rem;font-family:'Courier New',monospace;font-size:.85rem;font-weight:700;padding:.4rem .7rem;border-radius:3px;margin-top:.4rem;background:var(--d52-paper);border:1.5px solid var(--d52-green);color:var(--d52-green)}
.d52-caption{font-family:'Helvetica Neue',sans-serif;font-size:.78rem;color:var(--d52-muted);margin-top:.75rem;line-height:1.5}
</style>
<div class="d52-rtp">
  <div class="d52-node">
    <div class="d52-doc"><div class="d52-doc-title">Seed Doc</div><div class="d52-dline">2 cups flour</div><div class="d52-dline">1 cup sugar</div><div class="d52-dline">½ tsp salt</div><div class="d52-dline">— Mix dry</div><div class="d52-dline">— Bake 350°F</div></div>
    <div class="d52-node-lbl">Original <em>s</em></div>
  </div>
  <div class="d52-arr">
    <div class="d52-llm">Forward edit<br>"→ metric"<span class="d52-badge">🔄 Fresh context</span></div>
    <div class="d52-aline"></div><div class="d52-achev"></div>
  </div>
  <div class="d52-node">
    <div class="d52-doc"><div class="d52-doc-title">Transformed</div><div class="d52-dline">473 ml flour</div><div class="d52-dline">237 ml sugar</div><div class="d52-dline">2.5 ml salt</div><div class="d52-dline">— Mix dry</div><div class="d52-dline">— Bake 175°C</div></div>
    <div class="d52-node-lbl">Edited <em>t</em></div>
  </div>
  <div class="d52-arr">
    <div class="d52-llm">Backward edit<br>"→ imperial"<span class="d52-badge">🔄 Fresh context</span></div>
    <div class="d52-aline"></div><div class="d52-achev"></div>
  </div>
  <div class="d52-node">
    <div class="d52-doc corrupt"><div class="d52-doc-title">Reconstructed</div><div class="d52-dline cx">2.01 cups flour</div><div class="d52-dline">1 cup sugar</div><div class="d52-dline">½ tsp salt</div><div class="d52-dline">— Mix dry</div><div class="d52-dline">— Bake 350°F</div></div>
    <div class="d52-node-lbl">Reconstructed <em>ŝ</em></div>
  </div>
  <div class="d52-arr">
    <div class="d52-llm" style="background:var(--d52-green)">sim(<em>s</em>, <em>ŝ</em>)</div>
    <div class="d52-aline" style="background:var(--d52-green)"></div>
    <div class="d52-achev" style="border-top-color:var(--d52-green)"></div>
  </div>
  <div class="d52-node">
    <div class="d52-score">RS@2 = 97.3</div>
    <div class="d52-node-lbl">Reconstruction<br>Score</div>
  </div>
</div>
<div class="d52-caption">Each LLM call is independent with no conversation history. Errors survive into the next round because they are baked into the document itself, not the context window.</div>
</div>

It's worth checking out some examples in the GitHub repo, see [music](https://github.com/microsoft/delegate52/blob/main/domain_viewer/musicsheet.md#edit-tasks-6-total), [robotics](https://github.com/microsoft/delegate52/blob/main/domain_viewer/robotics.md#edit-tasks-6-total) and [Ham radio](https://github.com/microsoft/delegate52/blob/main/domain_viewer/hamradio.md#edit-tasks-6-total) as examples.

They also tested the inclusion of distractor documents in LLM interactions and found that they harm documents more as interaction length increases.

Basically, degradation severity is exacerbated by document size, interaction length, and the presence of distractor files. However, important to note that the LLM interactions themselves are stateless - it's not just that more noise in context causes outputs to degrade.

---

The simulation below steps through a recipe domain across 8 round-trips. Each forward edit converts imperial measurements to metric; each backward edit reverts. Watch how errors compound through the document across completely independent calls.

<div class="d52-sim-widget">
<style>
.d52-sim-widget{--d52s-ink:#1a1208;--d52s-paper:#f5f0e8;--d52s-warm:#ede8dc;--d52s-rule:#c8bfa8;--d52s-amber:#d4700a;--d52s-alight:#f5d49a;--d52s-apale:#fdf3dc;--d52s-green:#2d6a4f;--d52s-glight:#b7e4c7;--d52s-red:#9b2335;--d52s-rlight:#f4b8c1;--d52s-blue:#2563a8;--d52s-muted:#7a6e5f;font-family:Georgia,'Times New Roman',serif}
.d52-sim-widget *{box-sizing:border-box}
.d52s-controls{display:flex;align-items:center;gap:.85rem;margin-bottom:1.1rem;flex-wrap:wrap}
.d52s-btn{background:var(--d52s-ink);color:var(--d52s-paper);border:none;padding:.5rem 1.1rem;font-family:'Helvetica Neue',sans-serif;font-size:.8rem;font-weight:700;letter-spacing:.04em;cursor:pointer;border-radius:3px;transition:background .15s}
.d52s-btn:hover{background:var(--d52s-amber);color:var(--d52s-ink)}
.d52s-btn:disabled{opacity:.35;cursor:not-allowed;background:var(--d52s-ink);color:var(--d52s-paper)}
.d52s-btn.sec{background:transparent;color:var(--d52s-ink);border:1.5px solid var(--d52s-ink)}
.d52s-btn.sec:hover{background:var(--d52s-warm)}
.d52s-meta{font-family:'Courier New',monospace;font-size:.78rem;color:var(--d52s-muted)}
.d52s-score-lbl{font-family:'Courier New',monospace;font-size:.78rem;font-weight:700}
.d52s-istrip{background:var(--d52s-warm);border:1px solid var(--d52s-rule);border-radius:4px;padding:.7rem 1rem;margin-bottom:.85rem;display:grid;grid-template-columns:1fr 1fr;gap:.75rem;transition:opacity .3s}
.d52s-ibox{font-family:'Helvetica Neue',sans-serif;font-size:.78rem;line-height:1.4}
.d52s-ilbl{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--d52s-muted);display:flex;align-items:center;gap:.3rem;margin-bottom:.2rem}
.d52s-dot{width:6px;height:6px;border-radius:50%;display:inline-block}
.d52s-indicator{display:flex;align-items:center;gap:.5rem;font-family:'Helvetica Neue',sans-serif;font-size:.72rem;color:var(--d52s-muted);margin-bottom:.75rem;transition:opacity .3s}
.d52s-pill{background:var(--d52s-ink);color:var(--d52s-paper);padding:.15rem .5rem;border-radius:2px;font-size:.65rem;font-weight:700}
.d52s-fpill{background:var(--d52s-alight);color:var(--d52s-amber);padding:.15rem .5rem;border-radius:2px;font-size:.65rem;font-weight:700;border:1px solid var(--d52s-amber)}
.d52s-layout{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;margin-bottom:.75rem}
@media(max-width:560px){.d52s-layout{grid-template-columns:1fr}.d52s-istrip{grid-template-columns:1fr}}
.d52s-dlbl{font-family:'Helvetica Neue',sans-serif;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--d52s-muted);margin-bottom:.4rem}
.d52s-doc{background:#fff;border:1.5px solid var(--d52s-ink);border-radius:4px;padding:1rem;font-family:'Courier New',monospace;font-size:.75rem;line-height:1.8;min-height:190px;box-shadow:4px 4px 0 var(--d52s-rule)}
.d52s-doc .sh{font-family:'Helvetica Neue',sans-serif;font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin:.65rem 0 .25rem;color:var(--d52s-muted)}
.d52s-doc .ttl{font-family:Georgia,serif;font-size:.85rem;font-weight:700;margin-bottom:.4rem}
.d52s-il,.d52s-sl{display:flex;gap:.45rem;align-items:baseline;margin-bottom:.12rem;padding:.03rem .2rem;border-radius:2px}
.d52s-qty{min-width:52px;color:var(--d52s-blue)}
.ca{background:var(--d52s-alight) !important;color:var(--d52s-ink) !important}
.cr{background:var(--d52s-rlight) !important;color:var(--d52s-red) !important;text-decoration:line-through;opacity:.65}
.cn{background:var(--d52s-glight) !important;color:var(--d52s-green) !important}
.d52s-bottom{display:flex;align-items:flex-end;gap:1rem;margin-top:.5rem}
.d52s-bars{display:flex;gap:3px;align-items:flex-end;flex:1}
.d52s-bar{flex:1;background:var(--d52s-rule);border-radius:2px 2px 0 0;transition:height .4s ease,background .4s;min-height:3px}
.d52s-bar.act{background:var(--d52s-amber)}
.d52s-bar.pst{background:var(--d52s-muted)}
.d52s-big{font-family:'Courier New',monospace;font-size:2rem;font-weight:700;line-height:1;text-align:right;min-width:80px;transition:color .4s}
.d52s-biglbl{font-family:'Helvetica Neue',sans-serif;font-size:.68rem;color:var(--d52s-muted);text-align:right}
.d52s-legend{display:flex;gap:1.25rem;margin-top:.6rem;font-family:'Helvetica Neue',sans-serif;font-size:.73rem;flex-wrap:wrap}
.d52s-lswatch{padding:.1rem .35rem;border-radius:2px;font-size:.7rem}
@keyframes d52fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.d52s-doc{animation:d52fadeUp .3s ease}
</style>

<div class="d52s-controls">
  <button class="d52s-btn" id="d52s-next" onclick="d52sNext()">Next Round →</button>
  <button class="d52s-btn sec" onclick="d52sReset()">Reset</button>
  <span class="d52s-meta" id="d52s-rlbl">Round 0 of 8</span>
  <span class="d52s-score-lbl" id="d52s-slbl"></span>
</div>

<div class="d52s-istrip" id="d52s-istrip" style="opacity:.4">
  <div class="d52s-ibox"><div class="d52s-ilbl"><span class="d52s-dot" style="background:#2563a8"></span>Forward edit</div><span id="d52s-fwd">—</span></div>
  <div class="d52s-ibox"><div class="d52s-ilbl"><span class="d52s-dot" style="background:#2d6a4f"></span>Backward edit</div><span id="d52s-bwd">—</span></div>
</div>

<div class="d52s-indicator" id="d52s-ind" style="opacity:0">
  <span class="d52s-pill">LLM call</span>
  <span class="d52s-fpill">🔄 Fresh context — no history</span>
  <span>→ document received, instruction applied, output returned</span>
</div>

<div class="d52s-layout">
  <div>
    <div class="d52s-dlbl">Original (reference)</div>
    <div class="d52s-doc" id="d52s-orig">
      <div class="ttl">Chocolate Chip Cookies</div>
      <div class="sh">Ingredients</div>
      <div class="d52s-il"><span class="d52s-qty">2 cups</span><span>all-purpose flour</span></div>
      <div class="d52s-il"><span class="d52s-qty">1 cup</span><span>granulated sugar</span></div>
      <div class="d52s-il"><span class="d52s-qty">½ tsp</span><span>salt</span></div>
      <div class="d52s-il"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div>
      <div class="sh">Steps</div>
      <div class="d52s-sl">1. Cream butter and sugar until fluffy.</div>
      <div class="d52s-sl">2. Sift in flour and salt, fold gently.</div>
      <div class="d52s-sl">3. Bake at 350°F for 12 minutes.</div>
    </div>
  </div>
  <div>
    <div class="d52s-dlbl">After round-trip reconstruction</div>
    <div class="d52s-doc" id="d52s-cur">
      <div class="ttl">Chocolate Chip Cookies</div>
      <div class="sh">Ingredients</div>
      <div class="d52s-il"><span class="d52s-qty">2 cups</span><span>all-purpose flour</span></div>
      <div class="d52s-il"><span class="d52s-qty">1 cup</span><span>granulated sugar</span></div>
      <div class="d52s-il"><span class="d52s-qty">½ tsp</span><span>salt</span></div>
      <div class="d52s-il"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div>
      <div class="sh">Steps</div>
      <div class="d52s-sl">1. Cream butter and sugar until fluffy.</div>
      <div class="d52s-sl">2. Sift in flour and salt, fold gently.</div>
      <div class="d52s-sl">3. Bake at 350°F for 12 minutes.</div>
    </div>
  </div>
</div>

<div class="d52s-bottom">
  <div style="flex:1">
    <div class="d52s-bars" id="d52s-bars"></div>
    <div style="font-family:'Courier New',monospace;font-size:.63rem;color:var(--d52s-muted);text-align:right;margin-top:.2rem">RS@k across rounds</div>
  </div>
  <div>
    <div class="d52s-big" id="d52s-big" style="color:var(--d52s-ink)">—</div>
    <div class="d52s-biglbl">Reconstruction score</div>
  </div>
</div>

<div class="d52s-legend">
  <span><span class="d52s-lswatch ca">■</span> Corrupted value</span>
  <span><span class="d52s-lswatch cr">■</span> Deleted content</span>
  <span><span class="d52s-lswatch cn">■</span> Hallucinated content</span>
</div>

<script>
(function(){
var rounds=[
  {fwd:'Convert all measurements to metric units',bwd:'Convert all measurements back to imperial',score:97,doc:'<div class="ttl">Chocolate Chip Cookies</div><div class="sh">Ingredients</div><div class="d52s-il"><span class="d52s-qty ca">2.01 cups</span><span>all-purpose flour</span></div><div class="d52s-il"><span class="d52s-qty">1 cup</span><span>granulated sugar</span></div><div class="d52s-il"><span class="d52s-qty">½ tsp</span><span>salt</span></div><div class="d52s-il"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div><div class="sh">Steps</div><div class="d52s-sl">1. Cream butter and sugar until fluffy.</div><div class="d52s-sl">2. Sift in flour and salt, fold gently.</div><div class="d52s-sl">3. Bake at 350°F for 12 minutes.</div>'},
  {fwd:'Sort ingredients alphabetically',bwd:'Restore original ingredient order',score:94,doc:'<div class="ttl">Chocolate Chip Cookies</div><div class="sh">Ingredients</div><div class="d52s-il"><span class="d52s-qty ca">2.01 cups</span><span>all-purpose flour</span></div><div class="d52s-il ca"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div><div class="d52s-il ca"><span class="d52s-qty">1 cup</span><span>granulated sugar</span></div><div class="d52s-il"><span class="d52s-qty">½ tsp</span><span>salt</span></div><div class="sh">Steps</div><div class="d52s-sl">1. Cream butter and sugar until fluffy.</div><div class="d52s-sl">2. Sift in flour and salt, fold gently.</div><div class="d52s-sl">3. Bake at 350°F for 12 minutes.</div>'},
  {fwd:'Convert steps to passive voice',bwd:'Revert steps to active voice',score:89,doc:'<div class="ttl">Chocolate Chip Cookies</div><div class="sh">Ingredients</div><div class="d52s-il"><span class="d52s-qty ca">2.01 cups</span><span>all-purpose flour</span></div><div class="d52s-il"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div><div class="d52s-il"><span class="d52s-qty">1 cup</span><span>granulated sugar</span></div><div class="d52s-il"><span class="d52s-qty">½ tsp</span><span>salt</span></div><div class="sh">Steps</div><div class="d52s-sl ca">1. Cream butter and sugar until fluffy; sift in flour and salt.</div><div class="d52s-sl cr">2. Sift in flour and salt, fold gently.</div><div class="d52s-sl">3. Bake at 350°F for 12 minutes.</div>'},
  {fwd:'Add preparation notes after each step',bwd:'Remove preparation notes, restore original steps',score:81,doc:'<div class="ttl">Chocolate Chip Cookies</div><div class="sh">Ingredients</div><div class="d52s-il"><span class="d52s-qty ca">2.01 cups</span><span>all-purpose flour</span></div><div class="d52s-il"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div><div class="d52s-il"><span class="d52s-qty">1 cup</span><span>granulated sugar</span></div><div class="d52s-il ca"><span class="d52s-qty">1 tsp</span><span>salt</span></div><div class="sh">Steps</div><div class="d52s-sl ca">1. Cream butter and sugar until fluffy; sift in flour and salt.</div><div class="d52s-sl cr">2. Sift in flour and salt, fold gently.</div><div class="d52s-sl">3. Bake at 350°F for 12 minutes.</div>'},
  {fwd:'Rewrite recipe for a professional kitchen context',bwd:'Rewrite recipe for a home cook',score:72,doc:'<div class="ttl">Chocolate Chip Cookies</div><div class="sh">Ingredients</div><div class="d52s-il"><span class="d52s-qty ca">2.01 cups</span><span>all-purpose flour</span></div><div class="d52s-il"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div><div class="d52s-il"><span class="d52s-qty">1 cup</span><span>granulated sugar</span></div><div class="d52s-il ca"><span class="d52s-qty">1 tsp</span><span>salt</span></div><div class="sh">Steps</div><div class="d52s-sl ca">1. Cream butter and sugar until fluffy; sift in flour and salt.</div><div class="d52s-sl cr">2. Sift in flour and salt, fold gently.</div><div class="d52s-sl cr">3. Bake at 350°F for 12 minutes.</div>'},
  {fwd:'Group ingredients by category (dry/wet)',bwd:'Merge ingredient groups back to a flat list',score:61,doc:'<div class="ttl">Chocolate Chip Cookies</div><div class="sh">Ingredients</div><div class="d52s-il ca"><span class="d52s-qty">2.01 cups</span><span>all-purpose flour</span></div><div class="d52s-il ca"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div><div class="d52s-il ca"><span class="d52s-qty">1 cup</span><span>granulated sugar</span></div><div class="d52s-il ca"><span class="d52s-qty">1 tsp</span><span>salt</span></div><div class="d52s-il cn"><span class="d52s-qty">2 large</span><span>eggs (added by model)</span></div><div class="sh">Steps</div><div class="d52s-sl ca">1. Cream butter and sugar until fluffy; sift in flour and salt.</div><div class="d52s-sl cr">2. Sift in flour and salt, fold gently.</div><div class="d52s-sl cr">3. Bake at 350°F for 12 minutes.</div>'},
  {fwd:'Format recipe as a numbered list with time estimates',bwd:'Remove time estimates, restore original format',score:48,doc:'<div class="ttl">Chocolate Chip Cookies</div><div class="sh">Ingredients</div><div class="d52s-il ca"><span class="d52s-qty">2.01 cups</span><span>all-purpose flour</span></div><div class="d52s-il ca"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div><div class="d52s-il ca"><span class="d52s-qty">1 cup</span><span>granulated sugar</span></div><div class="d52s-il ca"><span class="d52s-qty">1 tsp</span><span>salt</span></div><div class="d52s-il cn"><span class="d52s-qty">2 large</span><span>eggs (added by model)</span></div><div class="sh">Steps</div><div class="d52s-sl ca">1. Cream butter, sugar, and eggs until fluffy.</div><div class="d52s-sl cn">2. Add vanilla extract and mix. (hallucinated)</div><div class="d52s-sl cr">3. Bake at 350°F for 12 minutes.</div>'},
  {fwd:'Translate recipe into French, then back to English',bwd:'Standardise units and terminology',score:31,doc:'<div class="ttl ca">Chocolate Chip Biscuits</div><div class="sh">Ingredients</div><div class="d52s-il ca"><span class="d52s-qty">480 ml</span><span>all-purpose flour</span></div><div class="d52s-il cr"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div><div class="d52s-il cn"><span class="d52s-qty">225 g</span><span>softened butter</span></div><div class="d52s-il ca"><span class="d52s-qty">1 cup</span><span>caster sugar</span></div><div class="d52s-il ca"><span class="d52s-qty">1 tsp</span><span>salt</span></div><div class="d52s-il cn"><span class="d52s-qty">2 large</span><span>eggs</span></div><div class="sh">Steps</div><div class="d52s-sl ca">1. Beat butter and sugar until pale and creamy.</div><div class="d52s-sl cn">2. Fold in sifted flour with a spatula.</div><div class="d52s-sl ca">3. Bake at 175°C for 10–15 min.</div>'}
];
var cur=0,max=rounds.length,scores=rounds.map(function(r){return r.score;});
var bars=document.getElementById('d52s-bars');
for(var i=0;i<max;i++){var b=document.createElement('div');b.className='d52s-bar';b.id='d52b'+i;b.style.height=Math.round((scores[i]/100)*60)+'px';bars.appendChild(b);}
function upBars(n){for(var i=0;i<max;i++){var b=document.getElementById('d52b'+i);if(!b)continue;b.classList.remove('act','pst');if(i===n-1)b.classList.add('act');else if(i<n-1)b.classList.add('pst');}}
function scoreColor(s){return s>=90?'#2d6a4f':s>=75?'#d4700a':s>=55?'#c05a00':'#9b2335';}
window.d52sNext=function(){
  if(cur>=max)return;
  var r=rounds[cur];cur++;
  document.getElementById('d52s-istrip').style.opacity='1';
  document.getElementById('d52s-fwd').textContent=r.fwd;
  document.getElementById('d52s-bwd').textContent=r.bwd;
  document.getElementById('d52s-ind').style.opacity='1';
  var doc=document.getElementById('d52s-cur');
  doc.style.animation='none';void doc.offsetWidth;doc.style.animation='d52fadeUp .3s ease';
  doc.innerHTML=r.doc;
  var c=scoreColor(r.score);
  document.getElementById('d52s-big').style.color=c;
  document.getElementById('d52s-big').textContent=r.score;
  document.getElementById('d52s-slbl').textContent='Round '+cur+' score: '+r.score;
  document.getElementById('d52s-slbl').style.color=c;
  document.getElementById('d52s-rlbl').textContent='Round '+cur+' of '+max;
  upBars(cur);
  if(cur>=max)document.getElementById('d52s-next').disabled=true;
};
window.d52sReset=function(){
  cur=0;
  document.getElementById('d52s-istrip').style.opacity='.4';
  document.getElementById('d52s-fwd').textContent='—';
  document.getElementById('d52s-bwd').textContent='—';
  document.getElementById('d52s-ind').style.opacity='0';
  document.getElementById('d52s-cur').innerHTML='<div class="ttl">Chocolate Chip Cookies</div><div class="sh">Ingredients</div><div class="d52s-il"><span class="d52s-qty">2 cups</span><span>all-purpose flour</span></div><div class="d52s-il"><span class="d52s-qty">1 cup</span><span>granulated sugar</span></div><div class="d52s-il"><span class="d52s-qty">½ tsp</span><span>salt</span></div><div class="d52s-il"><span class="d52s-qty">2 sticks</span><span>unsalted butter</span></div><div class="sh">Steps</div><div class="d52s-sl">1. Cream butter and sugar until fluffy.</div><div class="d52s-sl">2. Sift in flour and salt, fold gently.</div><div class="d52s-sl">3. Bake at 350°F for 12 minutes.</div>';
  document.getElementById('d52s-big').textContent='—';
  document.getElementById('d52s-big').style.color='var(--d52s-ink)';
  document.getElementById('d52s-slbl').textContent='';
  document.getElementById('d52s-rlbl').textContent='Round 0 of 8';
  document.getElementById('d52s-next').disabled=false;
  upBars(0);
};
})();
</script>
</div>

---
## DELEGATE-52

The benchmark contains 310 work environments across 52 domains. Each environment includes real seed documents, distractor files, and 5-10 reversible edit tasks that resemble the kinds of tasks a worker might delegate to an LLM.

![Grid of 52 domain icons organised into five colour-coded categories: Code and Configuration (11 domains including Python, Docker, JSON), Science and Engineering (11 domains including Crystal, Molecule, Quantum), Creative and Media (11 domains including Music Sheet, Screenplay, LaTeX), Structured Records (11 domains including Accounting, Genealogy, Spreadsheet), and Everyday (8 domains including Recipe, Chess, Transit).](../../_media/llms-corrupt-figure-3-categories.png)

*Figure 3 from [@labanLLMsCorruptYour2026] - the 52 domains across five categories: Code & Configuration, Science & Engineering, Creative & Media, Structured Records, and Everyday*

Figure 4 shows an example work environment from the accounting domain.

![Work environment diagram for the accounting domain, showing the Hack Club ledger as the seed document with distractor files including a chart of accounts and expense reimbursement policy. Ten edit tasks branch out, including category split, person split, CSV conversion, euro conversion, and fund accounting, each with a forward and backward instruction.](../../_media/llms-corrupt-figure-4-account-example.png)

*Figure 4 from [@labanLLMsCorruptYour2026] - a work environment from the accounting domain, using a Hack Club ledger as the seed document, with forward/backward edit pairs like splitting by expense category and merging back*

## Results

They tested 19 models across the benchmark. All 19 models degraded documents over the course of the simulation. The top performers, such as Gemini 3.1 Pro, Claude 4.6 Opus, and GPT 5.4, still corrupted an average of about 25% of the document content after 20 interactions. Across all tested models, average degradation was about 50%, with weaker models failing more severely.

![Heatmap table of round-trip relay scores for 19 LLMs at workflow lengths 2 through 20. All models show declining scores from left to right, colour-coded from green (high preservation) through yellow to red (severe degradation). Gemini 3.1 Pro scores highest at 80.9 after 20 interactions; GPT 5 Nano scores lowest at 10.0.](../../_media/llms-corrupt-table-1.png)

*Table 1 from [@labanLLMsCorruptYour2026] - round-trip relay results for 19 LLMs across 20 interactions, colour-coded by degradation severity. Every model declines over time; frontier models delay but do not avoid degradation.*

Short-term performance did not reliably predict long-horizon performance. Some models that looked similar after two interactions diverged sharply after twenty, while others that started behind later caught up. This is one of the reasons the paper argues for long-horizon evaluation rather than only testing one-shot or short workflows.

The kind of degradation also changes with model strength. Weaker models tend to lose content through deletion, while frontier models are more likely to preserve content but corrupt it.

## Takeaways

One takeaway is that we need to be careful not to extrapolate model capabilities from one area to all domains. Models follow a [Jagged Frontier of LLM Capability](../../permanent/jagged-frontier-of-llm-capability.md), where they can excel in some tasks while making serious errors in others. For example, they perform well on Python and poorly on some structured-but-unfamiliar document formats, such as textual 3D object files.

It also raises interesting questions about whether we need to decouple the reasoning engine from the state management system. LLMs may be useful as the reasoning layer, but long-running document workflows probably need external state, parsers, validators, diffs, tests, and reversible operations to prevent silent corruption.