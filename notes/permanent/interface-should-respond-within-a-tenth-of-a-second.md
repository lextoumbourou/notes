---
title: An Interface Should Respond Within A Tenth Of A Second
date: 2022-07-03 00:00
tags:
    - UserInterface
cover: /_media/loading-state-cover.png
hide_cover_in_article: true
summary: Developers must be vigilant of slow user interfaces
---

> "if your interface does not respond within a tenth of a second, the player will feel like something is wrong with the interface."
\- James Schell, The Art of Game Design: A Book of Lenses [^1]

According to [studies](https://www.nngroup.com/articles/response-times-3-important-limits) [^2], 0.1 seconds is roughly the response time limit for a user to feel like they are in control of an interface.

Up to 1 second is the cut-off for a user's flow to remain uninterrupted, though they will not feel in control.

Ten seconds is the hard limit for keeping a user's attention focused. After that, they will want to do other things while waiting for the interface to respond.

You can try it yourself. Each button here will change color when clicked and respond in the time shown in the button text.

<style>
.buttons {
    width: 100%;
    display: flex;
    flex-flow: wrap;
}

a.btn {
    padding: 0 1.25rem;
    line-height: 2.125rem;
    font-size: 0.875rem;
    font-weight: 400;
    text-align: center;
    margin: 0.5rem;
    border-radius: 0.5em;
    background-color: #7187A2;
    color: #fff;
    text-decoration: none;
    overflow: hidden;
    cursor: pointer;
    vertical-align: middle;
    outline: none;
    touch-action: none !important;
    -webkit-tap-highlight-color: rgba(0,0,0,0);
}

@keyframes pulse {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(0, 0, 0, 0.7);
    }

    70% {
        transform: scale(1);
        box-shadow: 0 0 0 10px rgba(0, 0, 0, 0);
    }

    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(0, 0, 0, 0);
    }
}
</style>

<script>
const colors = ["rgb(113, 135, 162)", "rgb(255, 105, 180)", "rgb(255, 0, 0)", "rgb(255, 142, 0)", "rgb(255, 209, 0)", "rgb(0, 142, 0)", "rgb(0, 192, 192)", "rgb(64, 0, 152)", "rgb(142, 0, 142)"];

// Thanks to https://stackoverflow.com/questions/34458815/comparing-rgb-colors-in-javascript
function rgbExtract(s) {
  var match = /^\s*rgb\(\s*(\d+),\s*(\d+),\s*(\d+)\)\s*$/.exec(s);
  if (match === null) {
    return null;
  }
  return {
    r: parseInt(match[1], 10),
    g: parseInt(match[2], 10),
    b: parseInt(match[3], 10)
  };
}

function rgbMatches(sText, tText) {
  var sColor = rgbExtract(sText),
    tColor = rgbExtract(tText);
  if (sColor === null || tColor === null) {
    return false;
  }
  var componentNames = ['r', 'g', 'b'];
  for (var i = 0; i < componentNames.length; ++i) {
    var name = componentNames[i];
    if (sColor[name] != tColor[name]) {
      return false;
    }
  }
  return true;
}

function toggleLoading(el, isOn) {
    if (isOn) {
        el.style.animation = "pulse 2s linear infinite"
    }
    else {
        el.style.removeProperty("animation");
    }
}

function changeColor(delay, id, loading) {
  var el = document.getElementById(id);

  if (loading) {
      toggleLoading(el, true)
  }

  setTimeout(() => {
    let color = window.getComputedStyle(el).getPropertyValue('background-color');
    var colorIndex = colors.findIndex(candidateColor => rgbMatches(candidateColor, color));
    var nextIndex = (colorIndex + 1) % colors.length;
    var nextColor = colors[nextIndex];
    el.style.backgroundColor = nextColor;
    toggleLoading(el, false)
  }, delay);
}
</script>

<div class="buttons">
    <a onclick="changeColor(50, this.id)" id="btn-1" class="btn"><span>0.05 secs</span></a>
    <a onclick="changeColor(100, this.id)" id="btn-2" class="btn"><span>0.1 secs</span></a>
    <a onclick="changeColor(500, this.id)" id="btn-3" class="btn"><span>0.5 secs</span></a>
    <a onclick="changeColor(1000, this.id)" id="btn-4" class="btn"><span>1 sec</span></a>
    <a onclick="changeColor(5000, this.id)" id="btn-5" class="btn"><span>5 secs</span></a>
    <a onclick="changeColor(10000, this.id)" id="btn-6" class="btn"><span>10 secs</span></a>
    <a onclick="changeColor(15000, this.id)" id="btn-7" class="btn"><span>15 secs</span></a>
</div>

Which of them makes you feel in control of the button color? Which of them feels like the computer is in control? Which of them makes you want to rage quit?

There are [many solutions](https://www.nngroup.com/articles/progress-indicators/) to make an interface feel responsive, even when a delay is required to return results: animations, loading spinners, progression indicators, skeleton objects, etc.

Here's one idea to make the buttons respond immediately using a fun pulsing animation I found [here](https://www.florin-pop.com/blog/2019/03/css-pulse-effect/).

<div class="buttons">
    <a onclick="changeColor(50, this.id, true)" id="btn-8" class="btn"><span>0.05 secs</span></a>
    <a onclick="changeColor(100, this.id, true)" id="btn-9" class="btn"><span>0.1 secs</span></a>
    <a onclick="changeColor(500, this.id, true)" id="btn-10" class="btn"><span>0.5 secs</span></a>
    <a onclick="changeColor(1000, this.id, true)" id="btn-11" class="btn"><span>1 sec</span></a>
    <a onclick="changeColor(5000, this.id, true)" id="btn-12" class="btn"><span>5 secs</span></a>
    <a onclick="changeColor(10000, this.id, true)" id="btn-13" class="btn"><span>10 secs</span></a>
    <a onclick="changeColor(15000, this.id, true)" id="btn-14" class="btn"><span>15 secs</span></a>
</div>

Notice how you still feel in control even at the longest wait time for a change.

Though the exact solution you choose will likely come from a designer (if you're lucky enough to work with one), a developer's responsibility is to understand which parts of an interface are likely to need these solutions.

Only we know which interactions can return results straight from the client, which need to request results from servers, which requests are produced quickly from a cache or will require expensive processing.

It is up to us to review designs and give feedback on these interface problem areas.

For the development of the Splash game, since the response time for any call to a server is difficult to estimate, we follow a simple rule:

> **Any button press that triggers a server invocation we must first acknowledge on the client.**

Usually we do this via a loading state. In code terms, we're doing this:

```lua
interface.onClick = function()
   toggleLoadingState(true)
   local response = invokeServer()
   displayResponse(response)
   toggleLoadingState(false)
end
```

For some particularly long-running requests, we aim to do them in the background to allow the player to continue to enjoy the game while they wait.

See also the [Goal Of A Game Interface](goal-of-a-game-interface.md).

[^1]: Schell, Jesse. The Art of Game Design: A Book of Lenses. Amsterdam, Boston; Elsevier/Morgan Kaufmann, 2008. (pg. 910)
[^2]: Nah, Fiona, "A Study on Tolerable Waiting Time: How Long Are Web Users Willing to Wait?" (2003). AMCIS 2003 Proceedings. 285.
http://aisel.aisnet.org/amcis2003/285

Photo by <a href="https://unsplash.com/@mike_van_den_bos?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Mike van den Bos</a> on <a href="https://unsplash.com/s/photos/loading?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
