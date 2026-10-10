/**
 * sketch.js: hand-drawn outlines, rules and highlighter swipes (rough.js),
 * plus the sketched neighbourhood graph in the article sidebar.
 *
 * Every decorated element keeps a plain CSS fallback (a border or rule);
 * once drawn, it gains `.is-sketched` and the CSS hides the fallback.
 */
(function () {
  "use strict";

  if (!window.rough) return;

  var SVG_NS = "http://www.w3.org/2000/svg";
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Generated content gets decorated by selector; templates use data-sketch.
  var AUTO = [
    [".prose-body div.highlight", "box"],
    [".prose-body > pre", "box"],
    [".prose-body .callout:not([data-callout='scribble'])", "box"],
  ];

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // Stable seed per element so a redraw never jitters into a new shape.
  function seedFor(el, i) {
    var s = (el.textContent || "").slice(0, 40) + i;
    var h = 0;
    for (var k = 0; k < s.length; k++) h = (h * 31 + s.charCodeAt(k)) | 0;
    return (Math.abs(h) % 2147483646) + 1;
  }

  function overlay(el) {
    var svg = el.querySelector(":scope > svg.sketch-layer");
    if (!svg) {
      svg = document.createElementNS(SVG_NS, "svg");
      svg.setAttribute("class", "sketch-layer");
      svg.setAttribute("aria-hidden", "true");
      svg.setAttribute("focusable", "false");
      el.appendChild(svg);
    }
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    return svg;
  }

  function drawOne(el, kind, i) {
    var w = el.offsetWidth;
    var h = el.offsetHeight;
    if (!w || !h) return;
    var svg = overlay(el);
    var rc = rough.svg(svg);
    var seed = seedFor(el, i);
    var ink = token("--ink");
    var pad = 3;
    var node;
    svg.setAttribute("width", w + pad * 2);
    svg.setAttribute("height", h + pad * 2);
    svg.style.left = -pad + "px";
    svg.style.top = -pad + "px";

    if (kind === "box") {
      node = rc.rectangle(pad + 1, pad + 1, w - 2, h - 2, {
        seed: seed, stroke: ink, strokeWidth: 1.6, roughness: 1.1, bowing: 0.8,
      });
    } else if (kind === "line") {
      // Full-bleed rules stay inside their box so the page never scrolls sideways.
      svg.setAttribute("width", w);
      svg.style.left = "0px";
      node = rc.line(0, pad + h - 1, w, pad + h - 1, {
        seed: seed, stroke: ink, strokeWidth: 2, roughness: 1.4, bowing: 0.6,
      });
    } else if (kind === "vline") {
      // A narrow layer inside the box: a scrolling sidebar must not gain a scrollbar from its own rule.
      svg.setAttribute("width", pad * 2 + 2);
      svg.setAttribute("height", h);
      svg.style.left = "0px";
      svg.style.top = "0px";
      node = rc.line(pad, 0, pad, h, {
        seed: seed, stroke: ink, strokeWidth: 1.6, roughness: 1.2, bowing: 0.4,
      });
    } else if (kind === "swipe" || kind === "underline") {
      // Sit under the last line of text, as wide as that line.
      var colour = token(el.dataset.sketchColor || "--hl-teal");
      var line = lastLine(el);
      var x0 = pad + 2 + line.left;
      var y = pad + line.bottom - (kind === "swipe" ? 3 : 2);
      node = rc.line(x0, y, x0 + Math.max(line.width - 4, 24), y - 2, {
        seed: seed, stroke: colour, strokeWidth: kind === "swipe" ? 7 : 4,
        roughness: 1.2, bowing: 1.4, disableMultiStroke: true,
      });
    }
    if (!node) return;
    svg.appendChild(node);
    el.classList.add("is-sketched");
    if (el.dataset.sketchDraw === "on" && !reduceMotion && !el.dataset.sketchDrawn) {
      el.dataset.sketchDrawn = "1";
      drawOn(node);
    }
  }

  // The last rendered line of an element's text, relative to the element.
  function lastLine(el) {
    var range = document.createRange();
    range.selectNodeContents(el);
    // Measure the text only, never our own drawing layer inside the element.
    var layer = el.querySelector(":scope > svg.sketch-layer");
    if (layer) range.setEndBefore(layer);
    var rects = Array.prototype.filter.call(range.getClientRects(), function (r) { return r.width > 1; });
    var box = el.getBoundingClientRect();
    if (!rects.length) return { left: 0, width: el.offsetWidth, bottom: el.offsetHeight };
    var bottom = Math.max.apply(null, rects.map(function (r) { return r.bottom; }));
    var onLast = rects.filter(function (r) { return r.bottom > bottom - 4; });
    var left = Math.min.apply(null, onLast.map(function (r) { return r.left; }));
    var right = Math.max.apply(null, onLast.map(function (r) { return r.right; }));
    return { left: left - box.left, width: right - left, bottom: el.offsetHeight };
  }

  // The one authored motion: the title's swipe is drawn on, once.
  function drawOn(group) {
    group.querySelectorAll("path").forEach(function (p) {
      var len = p.getTotalLength();
      p.style.strokeDasharray = len;
      p.style.strokeDashoffset = len;
      p.getBoundingClientRect();
      p.style.transition = "stroke-dashoffset 700ms cubic-bezier(0.16, 1, 0.3, 1) 150ms";
      p.style.strokeDashoffset = "0";
    });
  }

  var targets = [];

  function collect() {
    targets = [];
    document.querySelectorAll("[data-sketch]").forEach(function (el) {
      targets.push([el, el.dataset.sketch]);
    });
    AUTO.forEach(function (pair) {
      document.querySelectorAll(pair[0]).forEach(function (el) {
        targets.push([el, pair[1]]);
      });
    });
  }

  function redrawKind(kind) {
    targets.forEach(function (t, i) { if (t[1] === kind) drawOne(t[0], t[1], i); });
  }

  function drawAll() {
    targets.forEach(function (t, i) { drawOne(t[0], t[1], i); });
    drawGraphs();
    if (graphData) redrawKind("vline");
  }

  // ------------------------------------------------------------------
  // Sketched neighbourhood graph
  // ------------------------------------------------------------------

  var graphData = null;
  var MAX_NEIGHBOURS = 8;

  function neighbourhood(slug) {
    var names = {};
    graphData.nodes.forEach(function (n) { names[n.id] = n.text; });
    var seen = {};
    var list = [];
    graphData.links.forEach(function (l) {
      var other = null, dir = null;
      if (l.source === slug) { other = l.target; dir = "out"; }
      else if (l.target === slug) { other = l.source; dir = "in"; }
      if (!other || other === slug || !names[other] || other.indexOf("tag:") === 0) return;
      if (seen[other]) { seen[other].both = seen[other].dir !== dir || seen[other].both; return; }
      seen[other] = { id: other, text: names[other], dir: dir, both: false };
      list.push(seen[other]);
    });
    // Two-way links first: they are the note's closest neighbours.
    list.sort(function (a, b) { return (b.both ? 1 : 0) - (a.both ? 1 : 0); });
    return { title: names[slug], items: list };
  }

  function wrapLabel(text, maxChars) {
    var words = text.split(/\s+/);
    var lines = [""];
    words.forEach(function (word) {
      var cur = lines[lines.length - 1];
      if ((cur + " " + word).trim().length > maxChars && cur) lines.push(word);
      else lines[lines.length - 1] = (cur + " " + word).trim();
    });
    if (lines.length > 3) {
      lines = lines.slice(0, 3);
      lines[2] = lines[2] + "…";
    }
    return lines;
  }

  function arrowHead(rc, from, to, opts) {
    var ang = Math.atan2(to[1] - from[1], to[0] - from[0]);
    var len = 9;
    var a1 = [to[0] - len * Math.cos(ang - 0.45), to[1] - len * Math.sin(ang - 0.45)];
    var a2 = [to[0] - len * Math.cos(ang + 0.45), to[1] - len * Math.sin(ang + 0.45)];
    return rc.linearPath([a1, to, a2], opts);
  }

  // A softly rounded box, the way a marker goes round a corner.
  function softBox(rc, x, y, w, h, opts) {
    var r = Math.min(9, h / 3);
    var d = "M" + (x + r) + " " + y + " H" + (x + w - r) + " Q" + (x + w) + " " + y + " " + (x + w) + " " + (y + r) +
      " V" + (y + h - r) + " Q" + (x + w) + " " + (y + h) + " " + (x + w - r) + " " + (y + h) +
      " H" + (x + r) + " Q" + x + " " + (y + h) + " " + x + " " + (y + h - r) +
      " V" + (y + r) + " Q" + x + " " + y + " " + (x + r) + " " + y + " Z";
    return rc.path(d, opts);
  }

  // A doodled arrow: a gentle curve between two boxes, head at the target.
  function doodleEdge(rc, svg, from, to, bend, opts, headAtTo, headAtFrom) {
    var mx = (from[0] + to[0]) / 2, my = (from[1] + to[1]) / 2;
    var dx = to[0] - from[0], dy = to[1] - from[1];
    var len = Math.sqrt(dx * dx + dy * dy) || 1;
    var mid = [mx - (dy / len) * bend, my + (dx / len) * bend];
    svg.appendChild(rc.curve([from, mid, to], opts));
    if (headAtTo) svg.appendChild(arrowHead(rc, mid, to, opts));
    if (headAtFrom) svg.appendChild(arrowHead(rc, mid, from, opts));
  }

  function drawGraph(container) {
    var slug = container.dataset.slug;
    var hood = neighbourhood(slug);
    var w = container.clientWidth;
    if (!w) return;
    container.querySelectorAll("svg").forEach(function (s) { s.remove(); });

    var ink = token("--ink");
    var pink = token("--hl-pink");
    var paper = token("--paper");
    var items = hood.items.slice(0, MAX_NEIGHBOURS);
    var extra = hood.items.length - items.length;

    if (!items.length) {
      hideBlock(container);
      return;
    }

    // Stacked layout for a narrow column: notes linking here sit above the
    // current note, notes it links out to sit below, in rows that fit.
    var GAP_X = 12, GAP_Y = 34, CH = 8.4;

    function measure(lines) {
      var longest = lines.reduce(function (m, l) { return Math.max(m, l.length); }, 0);
      return { w: Math.max(64, longest * CH + 22), h: lines.length * 19 + 16 };
    }

    var twoUp = Math.floor(((w - GAP_X) / 2 - 22) / CH);
    var centreLines = wrapLabel(hood.title || slug, Math.max(12, Math.floor((w * 0.8 - 22) / CH)));
    var cm = measure(centreLines);
    var centre = { w: cm.w, h: cm.h, lines: centreLines };

    var TRUNK_GAP = 26;
    var halfW = (w - TRUNK_GAP) / 2;
    var pairChars = Math.floor((halfW - 22) / CH);

    function boxFor(item, chars, maxW) {
      var lines = wrapLabel(item.text, Math.max(10, chars));
      var m = measure(lines);
      return { item: item, w: Math.min(m.w, maxW), h: m.h, lines: lines };
    }

    // One row if it fits; otherwise pairs either side of a central trunk.
    function layoutSide(list) {
      var row = [], used = 0, fits = true;
      list.forEach(function (item) {
        var b = boxFor(item, twoUp, w);
        used += (row.length ? GAP_X : 0) + b.w;
        row.push(b);
      });
      if (used > w) fits = false;
      if (fits) return { trunk: false, rows: list.length ? [row] : [] };
      var rows = [];
      for (var i = 0; i < list.length; i += 2) {
        rows.push(list.slice(i, i + 2).map(function (item) { return boxFor(item, pairChars, halfW); }));
      }
      return { trunk: true, rows: rows };
    }

    var above = layoutSide(items.filter(function (it) { return it.dir === "in" || it.both; }));
    var below = layoutSide(items.filter(function (it) { return it.dir === "out" && !it.both; }));
    var boxes = [];
    var y = 6;

    function place(side, where) {
      side.rows.forEach(function (row) {
        var rowH = Math.max.apply(null, row.map(function (b) { return b.h; }));
        if (side.trunk) {
          // Left box hugs the trunk from the left, right box from the right.
          row.forEach(function (b, k) {
            b.cx = k === 0 ? w / 2 - TRUNK_GAP / 2 - b.w / 2 : w / 2 + TRUNK_GAP / 2 + b.w / 2;
            b.cy = y + rowH / 2;
            b.trunk = where;
            b.inner = k === 0 ? b.cx + b.w / 2 : b.cx - b.w / 2;
            boxes.push(b);
          });
        } else {
          var rowW = row.reduce(function (t, b) { return t + b.w; }, 0) + GAP_X * (row.length - 1);
          var x = (w - rowW) / 2;
          row.forEach(function (b) {
            b.cx = x + b.w / 2;
            b.cy = y + rowH / 2;
            x += b.w + GAP_X;
            boxes.push(b);
          });
        }
        y += rowH + (side.trunk ? 14 : GAP_Y);
      });
      if (side.trunk && side.rows.length) y += GAP_Y - 14;
    }

    place(above, "above");
    centre.cx = w / 2;
    centre.cy = y + centre.h / 2;
    y += centre.h + GAP_Y;
    place(below, "below");
    var h = y - GAP_Y + 6;

    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("width", w);
    svg.setAttribute("height", h);
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    svg.setAttribute("role", "group");
    svg.setAttribute("aria-label", "Notes connected to " + hood.title);
    container.appendChild(svg);
    var rc = rough.svg(svg);

    var edgeOpts = { stroke: ink, strokeWidth: 1.3, roughness: 1.2, bowing: 1.6 };
    ["above", "below"].forEach(function (where, t) {
      var group = boxes.filter(function (b) { return b.trunk === where; });
      if (!group.length) return;
      var o = Object.assign({ seed: 91 + t }, edgeOpts, { bowing: 0.5 });
      var ys = group.map(function (b) { return b.cy; });
      var far = where === "above" ? Math.min.apply(null, ys) : Math.max.apply(null, ys);
      var end = where === "above" ? centre.cy - centre.h / 2 - 3 : centre.cy + centre.h / 2 + 3;
      svg.appendChild(rc.line(w / 2, far, w / 2, end, o));
      // Links into this note arrive at it; links out leave it.
      if (where === "above") svg.appendChild(arrowHead(rc, [w / 2, far], [w / 2, end], o));
      group.forEach(function (b, i) {
        var bo = Object.assign({}, o, { seed: 101 + i + t * 20 });
        svg.appendChild(rc.line(b.inner + (b.inner < w / 2 ? 3 : -3), b.cy, w / 2, b.cy, bo));
        if (where === "below") {
          var tip = [b.inner + (b.inner < w / 2 ? 3 : -3), b.cy];
          svg.appendChild(arrowHead(rc, [w / 2, b.cy], tip, bo));
        }
      });
    });
    boxes.forEach(function (b, i) {
      if (b.trunk) return;
      var o = Object.assign({ seed: i + 7 }, edgeOpts);
      var up = b.cy < centre.cy;
      var spread = (b.cx - centre.cx) * 0.35;
      var p1 = [centre.cx + Math.max(-centre.w / 2 + 14, Math.min(centre.w / 2 - 14, spread)), up ? centre.cy - centre.h / 2 - 3 : centre.cy + centre.h / 2 + 3];
      var p2 = [b.cx, up ? b.cy + b.h / 2 + 3 : b.cy - b.h / 2 - 3];
      var bend = (i % 2 ? -1 : 1) * Math.min(14, Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 5);
      doodleEdge(rc, svg, p1, p2, bend, o,
        b.item.dir === "out" || b.item.both, b.item.dir === "in" || b.item.both);
    });

    function label(parent, box) {
      var text = document.createElementNS(SVG_NS, "text");
      text.setAttribute("class", "sg-label");
      text.setAttribute("text-anchor", "middle");
      box.lines.forEach(function (line, k) {
        var t = document.createElementNS(SVG_NS, "tspan");
        t.setAttribute("x", box.cx);
        t.setAttribute("y", box.cy - (box.lines.length - 1) * 9.5 + k * 19 + 5.5);
        t.textContent = line;
        text.appendChild(t);
      });
      parent.appendChild(text);
    }

    boxes.forEach(function (b, i) {
      var a = document.createElementNS(SVG_NS, "a");
      a.setAttribute("href", "/" + b.item.id + ".html");
      a.setAttribute("class", "sg-node");
      var title = document.createElementNS(SVG_NS, "title");
      title.textContent = b.item.text;
      a.appendChild(title);
      a.appendChild(softBox(rc, b.cx - b.w / 2, b.cy - b.h / 2, b.w, b.h, {
        seed: i + 31, stroke: ink, strokeWidth: 1.5, roughness: 1.1,
        fill: paper, fillStyle: "solid",
      }));
      label(a, b);
      svg.appendChild(a);
    });

    var cg = document.createElementNS(SVG_NS, "g");
    cg.setAttribute("class", "sg-node sg-current");
    cg.appendChild(softBox(rc, centre.cx - centre.w / 2, centre.cy - centre.h / 2, centre.w, centre.h, {
      seed: 3, stroke: ink, strokeWidth: 2, roughness: 1.1, fill: pink, fillStyle: "solid",
    }));
    label(cg, centre);
    svg.appendChild(cg);


    var more = container.parentNode.querySelector(".sg-more");
    if (more) {
      more.hidden = extra <= 0;
      if (extra > 0) more.querySelector("[data-count]").textContent = extra;
    }
  }

  // Hide the graph's block, and the whole sidebar when nothing else is in it.
  function hideBlock(container) {
    var block = container.closest(".side-block");
    block.hidden = true;
    var side = block.closest(".page-side");
    if (side && !side.querySelector(".side-block:not([hidden])")) side.hidden = true;
  }

  function drawGraphs() {
    var graphs = document.querySelectorAll(".sketch-graph[data-slug]");
    if (!graphs.length) return;
    if (!graphData) return;
    graphs.forEach(drawGraph);
  }

  function loadGraph() {
    var graphs = document.querySelectorAll(".sketch-graph[data-slug]");
    if (!graphs.length) return;
    fetch("/graph.json")
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (data) { graphData = data; drawGraphs(); redrawKind("vline"); })
      .catch(function () {
        graphs.forEach(hideBlock);
      });
  }

  // ------------------------------------------------------------------

  var pending;
  function redrawSoon() {
    clearTimeout(pending);
    pending = setTimeout(drawAll, 120);
  }

  function start() {
    collect();
    drawAll();
    loadGraph();
    if (window.ResizeObserver) {
      var lastWidth = window.innerWidth;
      new ResizeObserver(function () {
        if (window.innerWidth !== lastWidth) { lastWidth = window.innerWidth; redrawSoon(); }
      }).observe(document.body);
    }
    document.addEventListener("themechange", drawAll);
  }

  var ready = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { ready.then(start); });
  } else {
    ready.then(start);
  }
})();
