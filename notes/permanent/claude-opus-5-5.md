---
title: Claude Opus 5.5
slug: claude-opus-5-5
date: 2026-09-22 00:00
modified: 2026-09-23 08:03
category: model
summary: Anthropic's Opus 5.5 improves coding and computer use while cutting token prices. Includes a runnable example that turns this article into an N64-style quiz gameshow.
tags:
  - ModelRelease
  - Claude
notebook:
  timeout: 600000
  outputLimit: 200
---

**Claude Opus 5.5** is Anthropic's first model in the Claude 5.5 family, [released on 22 September 2026](https://www.anthropic.com/claude-opus-5-5). It improves on Opus 5 in coding, computer use and knowledge work, while reducing the price per token.

Anthropic claims it approaches [Claude Fable 5.1](https://www.anthropic.com/claude-fable-and-mythos-5-1) on most work, with lower running costs (fewer tokens per task) - apparently 40% cheaper than Opus 5 on typical workloads at default settings.

## Availability and context

As of **23 September 2026**, Opus 5.5 is available through Claude, the Claude API, AWS, Google Cloud and Microsoft Azure. The direct API identifier is `claude-opus-5-5`.

It accepts text and images, produces text, and supports tool use. The API lists a **1M-token context window** and **128K maximum output tokens**. See the [model documentation](https://platform.claude.com/docs/en/about-claude/models/overview).

Adaptive thinking is always enabled, with `medium` effort as the default. You can adjust the effort, but you can no longer switch to a separate non-thinking mode, bringing it in line with other reasoning models on the market. Not sure that's a good thing or bad thing, but it was always quite confusing that the Claude family of models had an effort setting that was separate from its reasoning capability.

## Benchmarks

These are results reported in Anthropic's [system card, Table 8.1.A](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf#page=174). I've added comparisons to other similar models, all based on the vendors' reports.

| Benchmark | Opus 5 | Opus 5.5 |
|---|---:|---:|
| [SWE-bench](swe-bench.md) Pro | 79.2% | 89.9% |
| [Terminal-Bench](terminal-bench.md) 4.0 | 52.3% | 66.4% |
| OSWorld 2.0, strict success | 37.2% | 48.7% |
| GDPval-AA v2.1, Elo | 1708 | 1846 |

The default evaluation uses adaptive thinking at `max` effort, averaged over five trials. Terminal-Bench 4.0 uses `xhigh`. These are higher effort settings than the API default. The launch results also include production safeguards, with fallback models handling some restricted tasks.

Opus 5.5 does not lead every comparison: the same table puts GPT-6 Astra ahead on Terminal-Bench-Science 0.1 and AutomationBench. Results from other developers use their published setups, so this is not a controlled comparison with an identical setup [Coding Harness](coding-harness.md).

The system card also reports a useful caveat: despite improved resistance to prompt injection overall, Opus 5.5 is more susceptible to malicious instructions hidden inside text a user pastes into their own message. Better benchmark results do not remove the need to distinguish instructions from untrusted content.

## Pricing and Cache Settings

Standard **Claude API prices**, checked on **23 September 2026**, in **USD per million tokens**:

| Usage | Price |
|---|---:|
| Uncached input | $4.00 |
| Cached input read | $0.20 |
| Cache write, 5 minutes | $5.00 |
| Cache write, 1 hour | $8.00 |
| Output, including thinking tokens | $20.00 |

Input and output are 20% cheaper than Opus 5; cache reads are 60% cheaper. Reusing a large context can therefore make a bigger difference than the headline input price suggests. [Pricing documentation](https://platform.claude.com/docs/en/about-claude/pricing).

Caching requires opting in through `cache_control`, either for automatic caching or explicit breakpoints. The minimum cacheable prefix is 512 tokens. The default lifetime is five minutes, refreshed when reused; a one-hour cache costs more to create. The example below makes one request without caching. [Prompt caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

<!-- notebook-input-end -->

## Real example

I used Opus 5.5 to turn this article into a 3D gameshow with an N64 look.

This uses the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) and my [Obsidian Markdown Notebook](obsidian-markdown-notebook-code-execution-with-outputs-stored-in-the-file.md) plugin. A small helper saves the HTML and token usage, then renders the game. The model call is just a normal SDK call, with a one-sentence system prompt.

If you want to run it yourself, download the [Notebook HTML helper](../_media/notesbylex_notebook_html-0.1.1-py3-none-any.whl), then install it in your notebook's Python environment:

```bash
pip install anthropic ./notesbylex_notebook_html-0.1.1-py3-none-any.whl
```

Set `ANTHROPIC_API_KEY` in the environment used to launch Obsidian. The example reads it from the notebook process's environment and passes it to the SDK, which sets the `X-Api-Key` header. Running the cell makes a paid request; the saved game plays without an API key or another model call.

```python {id=opus-55-quiz}
import os
from anthropic import Anthropic
from notebook_html import read_article, render_anthropic_html

api_key = os.environ["ANTHROPIC_API_KEY"]
article = read_article("claude-opus-5-5.md", before="Real example")
with Anthropic(api_key=api_key).messages.stream(
    model="claude-opus-5-5",
    max_tokens=128_000,
    output_config={"effort": "high"},
    system="Return a complete, self-contained HTML document.",
    messages=[{"role": "user", "content": f"""
Turn this article into a 3D gameshow that looks like an N64 game.
Use around 8 questions that test understanding of the article, with a short
explanation after each answer.
Support phones and keyboards. Use inline CSS and JavaScript, no external assets,
network requests or browser storage, and no em dashes. Keep the code compact,
with one shared question renderer.

Article:
{article}
"""}],
) as stream:
    response = stream.get_final_message()
render_anthropic_html(response, "../_media/claude-opus-5-5-quiz-high.html")
```
<!-- nb-output id="opus-55-quiz" hash="be11714f7c6cddef" format="html" -->
<div class="nb-output">
<div class="nb-output-html"><iframe title="Generated HTML" width="100%" height="900" style="border:0;display:block" sandbox="allow-scripts" srcdoc="&lt;!DOCTYPE html&gt;
&lt;html lang=&quot;en&quot;&gt;
&lt;head&gt;
&lt;meta charset=&quot;utf-8&quot;&gt;
&lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,initial-scale=1,viewport-fit=cover&quot;&gt;
&lt;title&gt;Opus 5.5 Quiz Show 64&lt;/title&gt;
&lt;style&gt;
*{box-sizing:border-box}
html,body{margin:0;height:100%;overflow:hidden;background:#1a0a34;color:#fff;font-family:&quot;Arial Black&quot;,&quot;Arial Bold&quot;,Impact,sans-serif;touch-action:manipulation;-webkit-tap-highlight-color:transparent}
canvas{position:fixed;inset:0;width:100%;height:100%;display:block}
#scan{position:fixed;inset:0;pointer-events:none;background:repeating-linear-gradient(0deg,#0000 0 2px,#0003 2px 3px),radial-gradient(#0000 60%,#0008)}
#fx{position:fixed;inset:0;pointer-events:none;opacity:0}
#ui{position:fixed;inset:0;margin:auto;max-width:880px;display:flex;flex-direction:column;gap:8px;overflow-y:auto;padding:max(10px,env(safe-area-inset-top)) 10px max(10px,env(safe-area-inset-bottom))}
#hud{display:flex;gap:8px;align-items:center}
.chip{font:inherit;font-size:14px;color:#fff;background:#000a;border:3px solid #fff;border-radius:99px;padding:4px 12px;text-shadow:2px 2px 0 #000}
#sc{color:#ffe14a;margin-left:auto}
#snd{cursor:pointer}
#gap{flex:1;min-height:30px}
.panel{background:linear-gradient(#3346e0ee,#141c78ee);border:4px solid #fff;border-radius:20px;padding:12px 16px;box-shadow:0 6px 0 #0009,inset 0 4px 0 #fff4,inset 0 -6px 0 #0004;text-shadow:2px 2px 0 #000}
.q{margin:0;font-size:clamp(17px,2.8vw,26px);line-height:1.25}
#fb p,.sub,.tip{font-family:&quot;Trebuchet MS&quot;,Verdana,sans-serif;font-weight:700;font-size:clamp(14px,2vw,18px);line-height:1.35;margin:.4em 0}
.tip{opacity:.85;font-size:clamp(12px,1.7vw,15px)}
#fb h2{margin:0;font-size:clamp(22px,4vw,34px);letter-spacing:2px}
.good{background:linear-gradient(#1fb34aee,#0b5e24ee)}
.bad{background:linear-gradient(#e8262aee,#6e0c10ee)}
#ans{display:grid;grid-template-columns:1fr 1fr;gap:8px}
#ans.one{grid-template-columns:1fr;max-width:420px;width:100%;margin:0 auto}
.b{display:flex;align-items:center;gap:10px;width:100%;font:inherit;font-size:clamp(14px,2vw,17px);line-height:1.2;color:#fff;text-align:left;text-shadow:2px 2px 0 #000;background:linear-gradient(#fff5,#fff0 45%,#0003),var(--c,#e8262a);border:3px solid #fff;border-radius:16px;padding:9px 12px;box-shadow:0 5px 0 #0009;cursor:pointer;transition:transform .12s}
.b:active{transform:translateY(3px);box-shadow:0 2px 0 #0009}
.k{flex:none;display:grid;place-items:center;width:36px;height:36px;border-radius:50%;font-size:19px;background:radial-gradient(circle at 35% 30%,#fff9,#fff0 45%),var(--c,#e8262a);border:3px solid #0007;box-shadow:inset 0 -3px 0 #0005}
.sel,.b:focus-visible{outline:4px solid #ffe14a;outline-offset:2px;transform:scale(1.02)}
.ok{--c:#1fb34a!important;animation:pop .5s}
.no{--c:#666!important;opacity:.85}
.gone{display:none}
.nx{margin-top:6px;justify-content:center;--c:#2a5cff}
.logo{text-align:center;font-size:clamp(40px,10vw,76px);line-height:.95;background:linear-gradient(#fffbd0,#ffd000 45%,#ff6a00 55%,#ffb000);-webkit-background-clip:text;background-clip:text;color:transparent;filter:drop-shadow(3px 4px 0 #000);text-shadow:none;letter-spacing:2px}
.logo small{display:block;font-size:.42em;letter-spacing:4px}
.big{text-align:center;font-size:30px;color:#ffe14a;margin:.2em 0;letter-spacing:3px}
.blink{text-align:center;color:#ffe14a;animation:bl 1s steps(2) infinite;margin:.3em 0 0}
@keyframes bl{50%{opacity:0}}
@keyframes pop{40%{transform:scale(1.07)}}
@media(max-width:620px){#ans{grid-template-columns:1fr}.b{padding:8px 10px}.k{width:30px;height:30px;font-size:16px}}
&lt;/style&gt;
&lt;/head&gt;
&lt;body&gt;
&lt;canvas id=&quot;c&quot; aria-hidden=&quot;true&quot;&gt;&lt;/canvas&gt;&lt;div id=&quot;scan&quot;&gt;&lt;/div&gt;&lt;div id=&quot;fx&quot;&gt;&lt;/div&gt;
&lt;main id=&quot;ui&quot;&gt;
  &lt;div id=&quot;hud&quot;&gt;&lt;span class=&quot;chip&quot; id=&quot;qn&quot;&gt;READY?&lt;/span&gt;&lt;span class=&quot;chip&quot; id=&quot;sc&quot;&gt;★ 0&lt;/span&gt;&lt;button class=&quot;chip&quot; id=&quot;snd&quot; aria-label=&quot;Toggle sound&quot;&gt;♪ ON&lt;/button&gt;&lt;/div&gt;
  &lt;section id=&quot;card&quot; class=&quot;panel&quot; aria-live=&quot;polite&quot;&gt;&lt;/section&gt;
  &lt;div id=&quot;gap&quot;&gt;&lt;/div&gt;
  &lt;section id=&quot;fb&quot; class=&quot;panel&quot; aria-live=&quot;polite&quot; hidden&gt;&lt;/section&gt;
  &lt;div id=&quot;ans&quot;&gt;&lt;/div&gt;
&lt;/main&gt;
&lt;script&gt;
const Q=[
{q:&quot;Anthropic says Opus 5.5 is about 40% cheaper than Opus 5 on typical workloads. Where does that saving come from?&quot;,
a:[&quot;Lower per-token prices plus fewer tokens used per task&quot;,&quot;Thinking tokens are no longer billed at all&quot;,&quot;A much smaller context window than Opus 5&quot;,&quot;Prompt caching is now free for every request&quot;],
x:&quot;Input and output prices fell 20%, and the model tends to finish tasks with fewer tokens. Thinking tokens are still billed at the $20 output rate.&quot;},
{q:&quot;What changed about thinking in Opus 5.5?&quot;,
a:[&quot;Adaptive thinking is always on; you tune effort but cannot turn it off&quot;,&quot;Thinking is off by default and must be switched on&quot;,&quot;Effort is locked at max and cannot be changed&quot;,&quot;Thinking only works when accessed through AWS&quot;],
x:&quot;Adaptive thinking is always enabled, with medium effort by default. You can adjust effort, but the separate non-thinking mode is gone.&quot;},
{q:&quot;Which limits does the API list for Opus 5.5?&quot;,
a:[&quot;1M-token context window and 128K max output tokens&quot;,&quot;128K context window and 1M max output tokens&quot;,&quot;200K context window and 64K max output tokens&quot;,&quot;1M context window and unlimited output tokens&quot;],
x:&quot;It takes up to 1M tokens of context and can write up to 128K output tokens. It accepts text and images, outputs text, and supports tool use.&quot;},
{q:&quot;You run Opus 5.5 at default API settings and score below the published benchmarks. What is the likeliest reason?&quot;,
a:[&quot;The benchmarks used max or xhigh effort, above the medium default&quot;,&quot;The benchmark table was actually measured on Opus 5&quot;,&quot;Default API settings switch off tool use&quot;,&quot;The published scores came from one lucky trial&quot;],
x:&quot;Reported results used adaptive thinking at max effort (xhigh for Terminal-Bench 4.0), averaged over five trials. The API default is medium, so default runs can score lower.&quot;},
{q:&quot;How should you read the benchmark comparisons with other developers&#x27; models?&quot;,
a:[&quot;Carefully: each vendor used its own setup, and Opus 5.5 does not win every test&quot;,&quot;As a controlled test with one identical coding harness&quot;,&quot;As proof that Opus 5.5 leads every single benchmark&quot;,&quot;As independent results from a neutral third party&quot;],
x:&quot;GPT-6 Astra leads on Terminal-Bench-Science 0.1 and AutomationBench, and rival scores use their own published setups, so it is not a like-for-like comparison.&quot;},
{q:&quot;What security caveat does the system card raise?&quot;,
a:[&quot;It is more easily fooled by malicious instructions hidden in text a user pastes into their own message&quot;,&quot;It can no longer be trusted to use tools at all&quot;,&quot;Its prompt injection resistance got worse in every scenario&quot;,&quot;It now ignores system prompts completely&quot;],
x:&quot;Overall prompt injection resistance improved, but pasted content inside the user&#x27;s own message is a weak spot. You still need to separate instructions from untrusted content.&quot;},
{q:&quot;Which price change makes reusing a large context especially attractive?&quot;,
a:[&quot;Cache reads are 60% cheaper, at $0.20 per million tokens&quot;,&quot;Output is 60% cheaper, at $8 per million tokens&quot;,&quot;Cache writes no longer cost anything&quot;,&quot;Uncached input dropped to $0.20 per million tokens&quot;],
x:&quot;Cached reads cost $0.20/M versus $4.00/M uncached. Input and output fell 20%, but cache reads fell 60%, so reused context saves more than the headline price suggests.&quot;},
{q:&quot;Which statement about prompt caching is true?&quot;,
a:[&quot;You opt in with cache_control, and the prefix must be at least 512 tokens&quot;,&quot;Caching happens automatically on every request with no setup&quot;,&quot;The default cache lasts one hour and never refreshes&quot;,&quot;A one-hour cache costs the same to create as a five-minute one&quot;],
x:&quot;Caching needs cache_control (automatic or explicit breakpoints). The default five-minute lifetime refreshes on reuse, and a one-hour write costs $8.00/M versus $5.00/M.&quot;}
];
const $=id=&gt;document.getElementById(id),cv=$(&#x27;c&#x27;),g=cv.getContext(&#x27;2d&#x27;),card=$(&#x27;card&#x27;),ans=$(&#x27;ans&#x27;),fbx=$(&#x27;fb&#x27;),fx=$(&#x27;fx&#x27;);
const PC=[[40,90,255],[30,180,70],[255,196,0],[232,38,42]],PX=[-4.5,-1.5,1.5,4.5],L=&#x27;ABCD&#x27;,FOG=[26,10,52],
PAL=[[255,70,170],[70,200,255],[255,225,60],[120,255,120]];
let W,H,FL,CX,CY,C,mode=&#x27;title&#x27;,qi=0,score=0,order=[],cor=0,pick=-1,sel=0,kb=false,evT=-9,evOk=0,qT=0,parts=[],cp=[0,7,18],ct=[0,2,-2],last=0,snd=1,AC,lock=0;
const P=[];
/* ---------- tiny 3D engine ---------- */
const sub=(a,b)=&gt;[a[0]-b[0],a[1]-b[1],a[2]-b[2]],dot=(a,b)=&gt;a[0]*b[0]+a[1]*b[1]+a[2]*b[2],
crs=(a,b)=&gt;[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],nrm=a=&gt;{const l=Math.hypot(...a)||1;return a.map(v=&gt;v/l)},
q=(v,c,o={})=&gt;P.push({v,c,...o}),
tf=(x,y,z,r=0)=&gt;{const c=Math.cos(r),n=Math.sin(r);return p=&gt;[x+p[0]*c+p[2]*n,y+p[1],z+p[2]*c-p[0]*n]};
function box(T,x,y,z,w,h,d,c,o){const v=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]].map(a=&gt;T([x+a[0]*w,y+a[1]*h,z+a[2]*d]));
[[0,3,2,1],[4,5,6,7],[0,4,7,3],[1,2,6,5],[3,7,6,2],[0,1,5,4]].forEach(f=&gt;q(f.map(i=&gt;v[i]),c,o))}
function cyl(T,x,y,z,r,h,n,cs,ctp,o){const b=[],t=[];for(let i=0;i&lt;n;i++){const a=i/n*6.283,cx=x+Math.cos(a)*r,cz=z+Math.sin(a)*r;b.push(T([cx,y,cz]));t.push(T([cx,y+h,cz]))}
for(let i=0;i&lt;n;i++){const j=(i+1)%n;q([b[i],t[i],t[j],b[j]],cs,o)}if(ctp)q(t.slice().reverse(),ctp,o)}
function disc(T,r,n,c,z=0,o={}){const v=[];for(let i=0;i&lt;n;i++){const a=i/n*6.283;v.push(T([Math.cos(a)*r,Math.sin(a)*r,z]))}q(v,c,{d:1,...o})}
function cone(ap,c,r,n,col){const b=[];for(let i=0;i&lt;n;i++){const a=i/n*6.283;b.push([c[0]+Math.cos(a)*r,c[1],c[2]+Math.sin(a)*r])}for(let i=0;i&lt;n;i++)q([ap,b[i],b[(i+1)%n]],col,{d:1,e:1,a:.13})}
function star(T,R,c){const F=[],B=[],h=.2;for(let i=0;i&lt;10;i++){const a=i/10*6.283+1.571,r=i%2?R*.45:R,x=Math.cos(a)*r,y=Math.sin(a)*r;F.push(T([x,y,h]));B.push(T([x,y,-h]))}
const cf=T([0,0,h*1.8]),cb=T([0,0,-h*1.8]);for(let i=0;i&lt;10;i++){const j=(i+1)%10;q([cf,F[i],F[j]],c);q([cb,B[j],B[i]],c);q([F[i],B[i],B[j],F[j]],c.map(v=&gt;v*.8))}
[-1,1].forEach(s=&gt;box(T,s*.2,.12,h*1.5,.06,.15,.05,[20,20,40]))}
function pr(p){const d=sub(p,C.p),z=dot(d,C.f);return z&gt;.3?[CX+dot(d,C.r)*FL/z,CY-dot(d,C.u)*FL/z,z]:0}
function draw(){const Lt=nrm([.4,.9,.6]),R=[];
for(const o of P){const v=o.v,n=nrm(crs(sub(v[1],v[0]),sub(v[2],v[0])));if(!o.d&amp;&amp;dot(n,sub(C.p,v[0]))&lt;=0)continue;
let zs=0,pts=[];for(const p of v){const a=pr(p);if(!a){pts=0;break}pts.push(a);zs+=a[2]}if(!pts)continue;zs/=v.length;
const l=o.e?1.1:.45+.7*Math.max(0,o.d?Math.abs(dot(n,Lt)):dot(n,Lt)),f=Math.min(1,Math.max(0,(zs-10)/24))*(o.e?.4:1);
R.push({z:zs+(o.b||0),pts,a:o.a||1,col:&#x27;rgb(&#x27;+o.c.map((c,i)=&gt;Math.min(255,c*l)*(1-f)+FOG[i]*f|0)+&#x27;)&#x27;})}
R.sort((a,b)=&gt;b.z-a.z);g.lineWidth=.7;g.lineJoin=&#x27;round&#x27;;
for(const r of R){g.globalAlpha=r.a;g.fillStyle=g.strokeStyle=r.col;g.beginPath();r.pts.forEach((p,i)=&gt;i?g.lineTo(p[0],p[1]):g.moveTo(p[0],p[1]));g.closePath();g.fill();if(r.a==1)g.stroke()}g.globalAlpha=1}
function txt(s,p,sz,col){const a=pr(p);if(!a)return;const px=sz*FL/a[2];g.font=`900 ${px}px &quot;Arial Black&quot;,Impact,sans-serif`;g.textAlign=&#x27;center&#x27;;g.textBaseline=&#x27;middle&#x27;;
g.lineWidth=Math.max(1,px/7);g.strokeStyle=&#x27;#000&#x27;;g.strokeText(s,a[0],a[1]);g.fillStyle=col;g.fillText(s,a[0],a[1])}
/* ---------- the studio ---------- */
function scene(t){P.length=0;const I=tf(0,0,0),fb=mode==&#x27;fb&#x27;,since=t-evT,cheer=fb&amp;&amp;evOk||mode==&#x27;end&#x27;&amp;&amp;score&gt;=Q.length/2,fl=fb&amp;&amp;since&lt;2;
for(let i=-6;i&lt;6;i++)for(let j=-5;j&lt;8;j++){const x=i*1.6,z=j*1.6;q([[x,0,z],[x,0,z+1.6],[x+1.6,0,z+1.6],[x+1.6,0,z]],(i+j)&amp;1?[80,44,160]:[34,18,92],{b:1e3})}
box(I,0,4,-7.6,10,4,.4,[70,36,130]);box(I,0,4.6,-7.1,4.2,2.1,.1,[12,18,70],{e:1});
for(let k=0;k&lt;24;k++){const u=k/6;let x,y;if(u&lt;1){x=-4.4+u*8.8;y=6.9}else if(u&lt;2){x=4.4;y=6.9-(u-1)*4.6}else if(u&lt;3){x=4.4-(u-2)*8.8;y=2.3}else{x=-4.4;y=2.3+(u-3)*4.6}
box(I,x,y,-7,.14,.14,.1,(k+(t*8|0))%3?[255,230,120]:[255,60,160],{e:1})}
for(const s of[-1,1])for(let k=0;k&lt;8;k++){const on=(k+(t*6|0))%2;box(I,s*6.6,.5+k,-6.4,.42,.42,.42,fl?(on?(evOk?[80,255,120]:[255,60,60]):[40,20,40]):PAL[(k+(t*3|0))%4],{e:on||!fl?1:0})}
cyl(I,0,0,-3,3.2,.3,12,[190,40,120],[255,190,60],{b:20});
for(const s of[-1,1])for(let r=0;r&lt;3;r++){const bx=s*(7.6+r*1.2);box(I,bx,(r+1)*.35,1.2,.6,(r+1)*.35,3.4,[90,70,150],{b:30});
for(let c=0;c&lt;4;c++){const z=-1.2+c*1.6,y=(r+1)*.7+(cheer?Math.abs(Math.sin(t*9+c*2+r))*.4:Math.sin(t*2+c+r)*.04);
box(I,bx,y+.4,z,.28,.4,.25,PAL[(c+r*2+(s&gt;0))%4]);box(I,bx,y+1.02,z,.22,.22,.22,[240,190,150])}}
const sp=fb?[PX[cor],1.4,1.5]:[0,.3,-3];for(const s of[-1,1]){const ap=[s*5+Math.sin(t+s)*.5,9.5,0];box(I,ap[0],ap[1]+.3,ap[2],.35,.35,.35,[60,60,75]);cone(ap,sp,1.3,8,[255,250,200])}
// host
const sh=fb&amp;&amp;!evOk&amp;&amp;since&lt;1.6?Math.sin(t*24)*.45*(1-since/1.6):0,jmp=cheer?Math.abs(Math.sin(t*7))*.4:0,arm=cheer?1:0,
talk=mode==&#x27;title&#x27;||mode==&#x27;q&#x27;&amp;&amp;t-qT&lt;2?Math.abs(Math.sin(t*16)):cheer?.8:0,
B=tf(0,.3+jmp+Math.sin(t*2)*.04,-3,Math.sin(t*.7)*.2+(fb?PX[cor]*.05:0)),Hd=p=&gt;B(tf(0,2.95,0,sh)(p));
[-1,1].forEach(s=&gt;{box(B,s*.3,.62,0,.22,.62,.24,[40,40,100]);box(B,s*.3,.08,.08,.24,.08,.32,[25,25,30]);
box(B,s*.87,1.75+arm*1.1,0,.19,.52,.19,[220,40,60]);box(B,s*.87,1.15+arm*2.3,0,.16,.16,.16,[255,200,150])});
box(B,0,1.8,0,.66,.6,.36,[220,40,60]);box(B,0,1.85,.37,.16,.52,.02,[245,245,245]);box(B,0,2.3,.4,.24,.08,.03,[255,210,0],{e:1});
if(!arm){box(B,.87,1.4,.25,.05,.2,.05,[70,70,80]);box(B,.87,1.66,.25,.1,.1,.1,[30,30,30])}
box(Hd,0,0,0,.46,.45,.42,[255,200,150]);box(Hd,0,.5,-.04,.5,.12,.48,[90,50,20]);
[-1,1].forEach(s=&gt;box(Hd,s*.18,.1,.43,.07,.11,.02,[20,20,30]));box(Hd,0,-.02,.46,.06,.08,.05,[240,170,130]);
box(Hd,0,-.22,.43,.17,.03+talk*.07,.02,[140,20,40]);
// podiums
PX.forEach((x,k)=&gt;{let top=PC[k],e=0,up=0;
if(fb){if(k==cor){e=1;top=(t*6|0)%2?[255,255,210]:[90,255,130];up=.15}else if(k==pick)top=[90,90,90];else top=top.map(v=&gt;v*.4)}
else if(mode==&#x27;q&#x27;&amp;&amp;kb&amp;&amp;k==sel){up=.1+Math.sin(t*6)*.05;e=1}
cyl(I,x,0,1.5,.85,1.3+up,8,[210,210,230],null);cyl(I,x,.5,1.5,.9,.3,8,top,null,{e});cyl(I,x,1.3+up,1.5,1,.12,8,top,top,{e});
if(fb&amp;&amp;k==cor)cyl(I,x,1.45,1.5,.8,8,8,[120,255,160],null,{e:1,a:.22,d:1})});
star(tf(0,9,-6.8,t*1.5),1,[255,215,0]);
for(let k=0;k&lt;6;k++){const a=k/6*6.283+t*.4,T=tf(Math.cos(a)*8,6+Math.sin(t*2+k)*.3,-3+Math.sin(a)*3.5,t*3+k);disc(T,.45,8,[255,190,0]);disc(T,.3,8,[255,235,120],.02);disc(T,.3,8,[255,235,120],-.02)}
parts.forEach(o=&gt;disc(tf(o.x,o.y,o.z,o.r),.12,4,o.c,0,{e:1}))}
function size(){const s=Math.sqrt(76800/(innerWidth*innerHeight));W=cv.width=Math.round(innerWidth*s);H=cv.height=Math.round(innerHeight*s);FL=Math.min(W*.85,H*1.1);CX=W/2;CY=H*(W&lt;H?.44:.48)}
function frame(ms){const t=ms/1000,dt=Math.min(.05,t-last);last=t;let p,l;
if(mode==&#x27;title&#x27;){const a=Math.sin(t*.2)*1.1;p=[Math.sin(a)*12,5.5+Math.sin(t*.4),-3+Math.cos(a)*13];l=[0,2.5,-3]}
else if(mode==&#x27;fb&#x27;){const x=PX[cor];p=[x*.4,3.6,9.5];l=[x*.5,1.8,-.5]}
else if(mode==&#x27;end&#x27;){p=[Math.sin(t*.4)*5,3.2,7.5];l=[0,2.8,-3]}
else{p=[Math.sin(t*.35)*1.2,4.2,11];l=[0,2,-1.5]}
if(W&lt;H){p[2]+=4;p[1]+=1.5}
const k=Math.min(1,dt*2.5);for(let i=0;i&lt;3;i++){cp[i]+=(p[i]-cp[i])*k;ct[i]+=(l[i]-ct[i])*k}
const f=nrm(sub(ct,cp)),r=nrm(crs(f,[0,1,0]));C={p:cp,f,r,u:crs(r,f)};
parts=parts.filter(o=&gt;{o.vy=Math.max(o.vy-5*dt,-2.2);o.x+=o.vx*dt;o.y+=o.vy*dt;o.z+=o.vz*dt;o.r+=dt*6;return o.y&gt;0});
scene(t);const gr=g.createLinearGradient(0,0,0,H);gr.addColorStop(0,&#x27;#05020f&#x27;);gr.addColorStop(1,&#x27;rgb(26,10,52)&#x27;);g.fillStyle=gr;g.fillRect(0,0,W,H);draw();
txt(mode==&#x27;title&#x27;?&#x27;OPUS 5.5&#x27;:mode==&#x27;end&#x27;?score+&#x27;/&#x27;+Q.length:&#x27;Q&#x27;+(qi+1),[0,4.6,-6.95],mode==&#x27;title&#x27;?1.1:2.1,&#x27;#ffe14a&#x27;);
if(mode!=&#x27;title&#x27;)PX.forEach((x,k)=&gt;txt(L[k],[x,1,2.45],.6,&#x27;#fff&#x27;));
requestAnimationFrame(frame)}
/* ---------- game flow ---------- */
function beep(f,d=.08,ty=&#x27;square&#x27;){if(!snd)return;try{AC=AC||new(window.AudioContext||window.webkitAudioContext);const n=AC.currentTime;
f.forEach((h,i)=&gt;{const o=AC.createOscillator(),v=AC.createGain(),s=n+i*d;o.type=ty;o.frequency.value=h;v.gain.setValueAtTime(.05,s);v.gain.exponentialRampToValueAtTime(.001,s+d*1.6);o.connect(v);v.connect(AC.destination);o.start(s);o.stop(s+d*1.7)})}catch(e){}}
function flash(c){fx.style.transition=&#x27;none&#x27;;fx.style.background=c;fx.style.opacity=.45;void fx.offsetWidth;fx.style.transition=&#x27;opacity .7s&#x27;;fx.style.opacity=0}
function confetti(){for(let i=0;i&lt;90;i++)parts.push({x:(Math.random()-.5)*14,y:8+Math.random()*5,z:-4+Math.random()*7,vx:Math.random()-.5,vy:0,vz:Math.random()-.5,r:Math.random()*6,c:PAL[i%4]})}
const act=fn=&gt;{const n=Date.now();if(n-lock&gt;250){lock=n;fn()}},now=()=&gt;performance.now()/1000;
function hi(){[...ans.children].forEach((b,i)=&gt;b.classList.toggle(&#x27;sel&#x27;,kb&amp;&amp;mode==&#x27;q&#x27;&amp;&amp;i==sel))}
function menu(html,label,fn){card.innerHTML=html;fbx.hidden=true;ans.className=&#x27;one&#x27;;ans.innerHTML=`&lt;button class=&quot;b&quot; id=&quot;go&quot;&gt;&lt;span class=&quot;k&quot;&gt;▶&lt;/span&gt;&lt;span&gt;${label}&lt;/span&gt;&lt;/button&gt;`;$(&#x27;go&#x27;).onclick=()=&gt;act(fn)}
function render(){const d=Q[qi],o=[0,1,2,3];for(let i=3;i&gt;0;i--){const j=Math.random()*(i+1)|0;[o[i],o[j]]=[o[j],o[i]]}
order=o;cor=o.indexOf(0);mode=&#x27;q&#x27;;pick=-1;sel=0;qT=now();$(&#x27;qn&#x27;).textContent=`QUESTION ${qi+1}/${Q.length}`;fbx.hidden=true;
card.innerHTML=`&lt;p class=&quot;q&quot;&gt;${d.q}&lt;/p&gt;`;ans.className=&#x27;&#x27;;
ans.innerHTML=o.map((a,k)=&gt;`&lt;button class=&quot;b&quot; data-k=&quot;${k}&quot; style=&quot;--c:rgb(${PC[k]})&quot;&gt;&lt;span class=&quot;k&quot;&gt;${L[k]}&lt;/span&gt;&lt;span&gt;${d.a[a]}&lt;/span&gt;&lt;/button&gt;`).join(&#x27;&#x27;);
hi();if(kb)ans.children[0].focus({preventScroll:true});beep([520,660],.07)}
function answer(k){if(mode!=&#x27;q&#x27;)return;pick=k;mode=&#x27;fb&#x27;;evT=now();evOk=k==cor;
if(evOk){score++;confetti();beep([523,659,784,1047],.08)}else beep([300,220,150],.14,&#x27;sawtooth&#x27;);
flash(evOk?&#x27;#3f6&#x27;:&#x27;#f33&#x27;);[...ans.children].forEach((b,i)=&gt;b.classList.add(i==cor?&#x27;ok&#x27;:i==k?&#x27;no&#x27;:&#x27;gone&#x27;));$(&#x27;sc&#x27;).textContent=&#x27;★ &#x27;+score;
fbx.className=&#x27;panel &#x27;+(evOk?&#x27;good&#x27;:&#x27;bad&#x27;);fbx.hidden=false;
fbx.innerHTML=`&lt;h2&gt;${evOk?&#x27;CORRECT!&#x27;:&#x27;WHOOPS!&#x27;}&lt;/h2&gt;&lt;p&gt;${Q[qi].x}&lt;/p&gt;&lt;button class=&quot;b nx&quot; id=&quot;nx&quot;&gt;${qi&lt;Q.length-1?&#x27;NEXT ▶&#x27;:&#x27;RESULTS ▶&#x27;}&lt;/button&gt;`;
$(&#x27;nx&#x27;).onclick=()=&gt;act(next);$(&#x27;nx&#x27;).focus({preventScroll:true})}
function next(){if(++qi&lt;Q.length)render();else end()}
function start(){qi=0;score=0;$(&#x27;sc&#x27;).textContent=&#x27;★ 0&#x27;;render()}
function end(){mode=&#x27;end&#x27;;$(&#x27;qn&#x27;).textContent=&#x27;RESULTS&#x27;;const n=Q.length,r=score==n?&#x27;SUPERSTAR!&#x27;:score&gt;=6?&#x27;GREAT RUN!&#x27;:score&gt;=4?&#x27;NOT BAD!&#x27;:&#x27;KEEP PRACTICING!&#x27;;
menu(`&lt;div class=&quot;logo&quot; style=&quot;font-size:clamp(32px,8vw,60px)&quot;&gt;${r}&lt;/div&gt;&lt;p class=&quot;big&quot;&gt;${&#x27;★&#x27;.repeat(score)}&lt;span style=&quot;opacity:.3&quot;&gt;${&#x27;★&#x27;.repeat(n-score)}&lt;/span&gt;&lt;/p&gt;&lt;p class=&quot;sub&quot;&gt;You scored ${score} out of ${n} on Claude Opus 5.5. ${score==n?&#x27;Flawless! You know your context windows from your cache reads.&#x27;:&#x27;Play again to collect every star.&#x27;}&lt;/p&gt;`,&#x27;PLAY AGAIN&#x27;,start);
if(score&gt;=n/2){confetti();beep([392,523,659,784,1047],.1)}else beep([392,330,262],.15,&#x27;triangle&#x27;)}
function title(){mode=&#x27;title&#x27;;$(&#x27;qn&#x27;).textContent=&#x27;READY?&#x27;;
menu(`&lt;div class=&quot;logo&quot;&gt;OPUS 5.5&lt;small&gt;QUIZ SHOW 64&lt;/small&gt;&lt;/div&gt;&lt;p class=&quot;sub&quot;&gt;${Q.length} questions about Anthropic&#x27;s Claude Opus 5.5. Pick the right podium and collect stars!&lt;/p&gt;&lt;p class=&quot;tip&quot;&gt;Keyboard: 1-4 or A-D to answer, arrows + Enter to choose, M toggles sound. Phone: just tap.&lt;/p&gt;&lt;p class=&quot;blink&quot;&gt;PRESS START&lt;/p&gt;`,&#x27;START&#x27;,start)}
function toggleSound(){snd=!snd;$(&#x27;snd&#x27;).textContent=&#x27;♪ &#x27;+(snd?&#x27;ON&#x27;:&#x27;OFF&#x27;)}
ans.onclick=e=&gt;{const b=e.target.closest(&#x27;[data-k]&#x27;);if(b)act(()=&gt;answer(+b.dataset.k))};
ans.addEventListener(&#x27;focusin&#x27;,e=&gt;{const b=e.target.closest(&#x27;[data-k]&#x27;);if(b){sel=+b.dataset.k;hi()}});
$(&#x27;snd&#x27;).onclick=toggleSound;
addEventListener(&#x27;pointerdown&#x27;,()=&gt;{kb=false;hi()});
addEventListener(&#x27;keydown&#x27;,e=&gt;{const k=e.key.toLowerCase();if(k==&#x27;m&#x27;)return toggleSound();if(k==&#x27;tab&#x27;){kb=true;return}
let n=&#x27;1234&#x27;.indexOf(k);if(n&lt;0)n=&#x27;abcd&#x27;.indexOf(k);
if(mode==&#x27;q&#x27;){if(n&gt;=0&amp;&amp;k.length==1){e.preventDefault();kb=true;return act(()=&gt;answer(n))}
const m={arrowleft:-1,arrowright:1,arrowup:-2,arrowdown:2}[k];
if(m){e.preventDefault();kb=true;sel=(sel+m+4)%4;ans.children[sel].focus({preventScroll:true});hi();beep([440],.03);return}
if(k==&#x27;enter&#x27;||k==&#x27; &#x27;){e.preventDefault();kb=true;act(()=&gt;answer(sel))}}
else if(k==&#x27;enter&#x27;||k==&#x27; &#x27;){e.preventDefault();kb=true;act(mode==&#x27;fb&#x27;?next:start)}});
addEventListener(&#x27;resize&#x27;,size);size();title();requestAnimationFrame(frame);
&lt;/script&gt;
&lt;/body&gt;
&lt;/html&gt;"></iframe></div>
</div>
<!-- /nb-output -->

This game was generated on **23 September 2026** using **high effort**. It cost about **US$0.96**, calculated from the [saved token usage](../_media/claude-opus-5-5-quiz-high.json) and [standard API prices](https://platform.claude.com/docs/en/about-claude/pricing):

| Usage | Tokens | Estimated cost (USD) |
|---|---:|---:|
| Input | 1,594 | $0.0064 |
| Output, including thinking | 47,542 | $0.9508 |
| **Total** | **49,136** | **$0.9572** |

Of those output tokens, **36,150 were thinking tokens** and **11,392 were the HTML response**. Thinking is included in the output cost.

[Play the quiz in its own page](../_media/claude-opus-5-5-quiz-high.html).

The example sets [effort](https://platform.claude.com/docs/en/build-with-claude/effort) to `high`. To render the saved game again without calling the model, use `load_html("../_media/claude-opus-5-5-quiz-high.html")` from `notebook_html`.

## Sources

- [Claude Opus 5.5 announcement](https://www.anthropic.com/claude-opus-5-5)
- [Claude Opus 5.5 system card](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf)
- [Model specifications](https://platform.claude.com/docs/en/about-claude/models/overview)
- [API pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Steering thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)
