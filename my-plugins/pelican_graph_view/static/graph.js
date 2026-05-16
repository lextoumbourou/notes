/**
 * pelican_graph_view — graph.js
 *
 * D3 (force simulation) + PixiJS 7 (WebGL rendering) interactive graph view.
 * Inspired by the Quartz graph implementation.
 *
 * CDN dependencies (injected by plugin or graph_page.html):
 *   D3 v7      — https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js
 *   PixiJS v7  — https://cdn.jsdelivr.net/npm/pixi.js@7/dist/pixi.min.js
 *   Tween.js   — https://cdn.jsdelivr.net/npm/@tweenjs/tween.js@23/dist/tween.umd.js
 */

(function () {
  "use strict";

  // ------------------------------------------------------------------
  // localStorage — visited nodes
  // ------------------------------------------------------------------

  const LOCAL_STORAGE_KEY = "graph-visited";

  function getVisited() {
    try {
      return new Set(JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY) || "[]"));
    } catch {
      return new Set();
    }
  }

  function addToVisited(slug) {
    const visited = getVisited();
    visited.add(slug);
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify([...visited]));
    } catch {
      // ignore storage quota / private-mode errors
    }
  }

  // ------------------------------------------------------------------
  // Color utilities
  // ------------------------------------------------------------------

  /**
   * Convert a CSS color string to a PixiJS numeric color (0xRRGGBB).
   * Handles: #rgb, #rrggbb, rgb(...), rgba(...), and named colors via a
   * temporary canvas trick. Falls back to 0x888888 on failure.
   */
  const _colorCache = new Map();

  function cssColorToNumber(cssColor) {
    if (!cssColor) return 0x888888;
    const key = cssColor.trim();
    if (_colorCache.has(key)) return _colorCache.get(key);

    let num = 0x888888;
    try {
      if (/^#([0-9a-f]{3}){1,2}$/i.test(key)) {
        // Hex shorthand or longhand
        let hex = key.slice(1);
        if (hex.length === 3) {
          hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
        }
        num = parseInt(hex, 16);
      } else {
        // Use an off-screen canvas to resolve rgb(...) and named colors
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 1;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = key;
        ctx.fillRect(0, 0, 1, 1);
        const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
        num = (r << 16) | (g << 8) | b;
      }
    } catch {
      // ignore parse errors
    }

    _colorCache.set(key, num);
    return num;
  }

  function getCSSVar(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();
  }

  function resolveColor(name, fallback) {
    const val = getCSSVar(name);
    return val || fallback;
  }

  // ------------------------------------------------------------------
  // BFS neighbourhood filter
  // ------------------------------------------------------------------

  /**
   * Return the Set of node ids reachable within `depth` undirected hops
   * from `startId`. Returns null to mean "all nodes" (depth < 0).
   */
  function bfsNeighbourhood(startId, links, depth) {
    if (depth < 0) return null;

    // Build undirected adjacency list
    const adj = new Map();
    for (const lnk of links) {
      if (!adj.has(lnk.source)) adj.set(lnk.source, []);
      if (!adj.has(lnk.target)) adj.set(lnk.target, []);
      adj.get(lnk.source).push(lnk.target);
      adj.get(lnk.target).push(lnk.source);
    }

    const neighbourhood = new Set([startId]);
    const queue = [startId, "__SENTINEL"];
    let remaining = depth;

    while (remaining >= 0 && queue.length > 0) {
      const cur = queue.shift();
      if (cur === "__SENTINEL") {
        remaining--;
        if (queue.length > 0) queue.push("__SENTINEL");
      } else {
        for (const neighbour of adj.get(cur) || []) {
          if (!neighbourhood.has(neighbour)) {
            neighbourhood.add(neighbour);
            queue.push(neighbour);
          }
        }
      }
    }

    return neighbourhood;
  }

  // ------------------------------------------------------------------
  // Navigation
  // ------------------------------------------------------------------

  /**
   * Navigate to a node's article page.
   * Uses an absolute URL from the site root to avoid path-relative issues
   * when RELATIVE_URLS = True and pages are nested.
   */
  function navigateToSlug(slug) {
    if (slug.startsWith("tag:")) return; // tag pseudo-nodes have no page
    window.location.href = `${window.location.origin}/${slug}.html`;
  }

  // ------------------------------------------------------------------
  // Core render function
  // ------------------------------------------------------------------

  /**
   * Render a force-directed graph into `container`.
   *
   * @param {HTMLElement} container   — DOM element that receives the <canvas>
   * @param {string|null} focusSlug  — current article slug (null for global view)
   * @param {object[]}    allNodes   — full node list from graph.json
   * @param {object[]}    allLinks   — full link list from graph.json
   * @param {object}      cfg        — merged config object (from data-cfg)
   * @returns {Function}             — cleanup() to destroy the graph
   */
  async function renderGraph(container, focusSlug, allNodes, allLinks, cfg) {
    // Clear previous content
    while (container.firstChild) container.removeChild(container.firstChild);

    // ---- Config -------------------------------------------------------
    const depth          = cfg.depth          != null ? cfg.depth          : (focusSlug ? 1 : -1);
    const showTags       = cfg.showTags       !== false;
    const focusOnHover   = cfg.focusOnHover   === true;
    const enableRadial   = cfg.enableRadial   === true;
    const repelForce     = cfg.repelForce     != null ? cfg.repelForce     : 0.5;
    const centerForce    = cfg.centerForce    != null ? cfg.centerForce    : 0.3;
    const linkDistance   = cfg.linkDistance   != null ? cfg.linkDistance   : 30;
    const fontSize       = cfg.fontSize       != null ? cfg.fontSize       : 0.6;
    const scaleByDegree  = cfg.scaleNodesByDegree !== false;

    const visited = getVisited();

    // ---- Resolve colors from CSS custom properties -------------------
    const colors = {
      secondary:  resolveColor("--g-node-self", resolveColor("--secondary",  "#B58B3C")),
      tertiary:   resolveColor("--g-node",      resolveColor("--tertiary",   "#3B6A8A")),
      gray:       resolveColor("--g-muted",     resolveColor("--gray",       "#9b9b9b")),
      light:      resolveColor("--g-surface",   resolveColor("--light",      "#e8e8e8")),
      lightgray:  resolveColor("--g-rule",      resolveColor("--lightgray",  "#d4d4d4")),
      dark:       resolveColor("--g-ink",       resolveColor("--dark",       "#141021")),
      bodyFont:   resolveColor("--font-sans",   resolveColor("--bodyFont",   "sans-serif")),
    };

    function nodeColor(node) {
      if (node.id === focusSlug)                                    return colors.secondary;
      if (visited.has(node.id) || node.id.startsWith("tag:"))      return colors.tertiary;
      return colors.gray;
    }

    // ---- Filter links and nodes to neighbourhood --------------------
    let links = showTags
      ? allLinks
      : allLinks.filter(
          (l) => !l.source.startsWith("tag:") && !l.target.startsWith("tag:")
        );

    const nodeIds =
      focusSlug && depth >= 0
        ? bfsNeighbourhood(focusSlug, links, depth)
        : null; // null = show all

    const filteredNodes = allNodes.filter((n) => {
      if (!showTags && n.id.startsWith("tag:")) return false;
      if (nodeIds !== null && !nodeIds.has(n.id)) return false;
      return true;
    });

    const nodeIdSet = new Set(filteredNodes.map((n) => n.id));

    // Update node count badge if present (local graph panel)
    if (focusSlug) {
      const card = container.closest(".panel-card");
      if (card) {
        const badge = card.querySelector(".graph-node-count");
        if (badge) badge.textContent = `${filteredNodes.length} nodes`;
      }
    }

    const filteredLinks = links.filter(
      (l) => nodeIdSet.has(l.source) && nodeIdSet.has(l.target)
    );

    // ---- Degree map for node sizing ---------------------------------
    const degreeMap = new Map(filteredNodes.map((n) => [n.id, 0]));
    for (const l of filteredLinks) {
      degreeMap.set(l.source, (degreeMap.get(l.source) || 0) + 1);
      degreeMap.set(l.target, (degreeMap.get(l.target) || 0) + 1);
    }

    function nodeRadius(id) {
      if (!scaleByDegree) return 4;
      return 2 + Math.sqrt(degreeMap.get(id) || 0);
    }

    // ---- Canvas dimensions ------------------------------------------
    const width  = container.offsetWidth  || 400;
    const height = Math.max(container.offsetHeight || 300, 200);

    // ---- PixiJS setup -----------------------------------------------
    const app = new PIXI.Application({
      width,
      height,
      antialias:       true,
      autoStart:       false,
      autoDensity:     true,
      backgroundAlpha: 0,
      resolution:      window.devicePixelRatio || 1,
    });

    container.appendChild(app.view);

    const stage = app.stage;
    stage.sortableChildren = true;

    const linkContainer  = new PIXI.Container(); linkContainer.zIndex  = 1;
    const nodesContainer = new PIXI.Container(); nodesContainer.zIndex = 2;
    const labelsContainer= new PIXI.Container(); labelsContainer.zIndex= 3;
    stage.addChild(linkContainer, nodesContainer, labelsContainer);

    // ---- D3 simulation data -----------------------------------------
    // D3 mutates these objects in-place; start with plain id strings
    const simNodes = filteredNodes.map((n) => ({ ...n }));
    const simLinks = filteredLinks.map((l) => ({ source: l.source, target: l.target }));

    // ---- D3 force simulation ----------------------------------------
    const simulation = d3
      .forceSimulation(simNodes)
      .force("charge", d3.forceManyBody().strength(-100 * repelForce))
      .force("center",  d3.forceCenter().strength(centerForce))
      .force("link",
        d3.forceLink(simLinks).id((d) => d.id).distance(linkDistance)
      )
      .force("collide",
        d3.forceCollide((d) => nodeRadius(d.id)).iterations(3)
      );

    if (enableRadial) {
      const radius = (Math.min(width, height) / 2) * 0.8;
      simulation.force("radial", d3.forceRadial(radius).strength(0.2));
    }

    // ---- Build PixiJS render objects --------------------------------

    // Link graphics (one Graphics per link)
    const linkRenderData = simLinks.map(() => {
      const gfx = new PIXI.Graphics();
      linkContainer.addChild(gfx);
      return { gfx, alpha: 1, color: colors.lightgray, active: false };
    });

    // Node graphics + text labels
    const nodeRenderData = simNodes.map((node) => {
      const r     = nodeRadius(node.id);
      const isTag = node.id.startsWith("tag:");
      const fill  = cssColorToNumber(isTag ? colors.light : nodeColor(node));

      const gfx = new PIXI.Graphics();
      gfx.beginFill(fill);
      gfx.drawCircle(0, 0, r);
      if (isTag) {
        gfx.endFill();
        gfx.lineStyle(2, cssColorToNumber(colors.tertiary));
        gfx.drawCircle(0, 0, r);
      }
      gfx.endFill();

      gfx.interactive = true;
      gfx.buttonMode  = true;
      gfx.cursor      = "pointer";

      const label = new PIXI.Text(node.text, {
        fontSize:   fontSize * 15,
        fill:       colors.dark,
        fontFamily: colors.bodyFont,
      });
      label.anchor.set(0.5, 1.2);
      label.alpha      = node.id === focusSlug ? 1 : 0;
      label.resolution = (window.devicePixelRatio || 1) * 2;

      nodesContainer.addChild(gfx);
      labelsContainer.addChild(label);

      return { node, gfx, label, alpha: 1, active: false };
    });

    // Quick lookup: id → render datum
    const nodeById = new Map(nodeRenderData.map((rd) => [rd.node.id, rd]));

    // ---- Hover state ------------------------------------------------

    let hoveredId = null;

    function updateHover(newId) {
      hoveredId = newId;

      if (newId === null) {
        for (const rd of nodeRenderData)  rd.active = false;
        for (const ld of linkRenderData)  ld.active = false;
        return;
      }

      const neighbours = new Set();
      simLinks.forEach((l, i) => {
        const srcId = typeof l.source === "object" ? l.source.id : l.source;
        const tgtId = typeof l.target === "object" ? l.target.id : l.target;
        const isActive = srcId === newId || tgtId === newId;
        linkRenderData[i].active = isActive;
        if (isActive) { neighbours.add(srcId); neighbours.add(tgtId); }
      });

      for (const rd of nodeRenderData) {
        rd.active = neighbours.has(rd.node.id);
      }
    }

    // ---- Tween helpers ----------------------------------------------

    function tweenTo(obj, props, duration) {
      return new TWEEN.Tween(obj).to(props, duration).start();
    }

    function renderLinks() {
      for (const ld of linkRenderData) {
        const targetAlpha = hoveredId ? (ld.active ? 1 : 0.2) : 1;
        ld.color = ld.active ? colors.gray : colors.lightgray;
        tweenTo(ld, { alpha: targetAlpha }, 200);
      }
    }

    function renderNodes() {
      for (const rd of nodeRenderData) {
        const targetAlpha =
          hoveredId !== null && focusOnHover ? (rd.active ? 1 : 0.2) : 1;
        tweenTo(rd, { alpha: targetAlpha }, 200);
      }
    }

    function renderLabels() {
      for (const rd of nodeRenderData) {
        if (hoveredId === rd.node.id) {
          tweenTo(rd.label, { alpha: 1 }, 100);
        } else if (rd.node.id !== focusSlug) {
          tweenTo(rd.label, { alpha: 0 }, 200);
        }
      }
    }

    function renderAll() {
      renderNodes();
      renderLinks();
      renderLabels();
    }

    // ---- Node interaction -------------------------------------------

    let dragging      = false;
    let dragStartTime = 0;

    for (const rd of nodeRenderData) {
      rd.gfx
        .on("pointerover", () => {
          updateHover(rd.node.id);
          rd.label.alpha = 1;
          renderAll();
        })
        .on("pointerout", () => {
          updateHover(null);
          renderAll();
        })
        .on("click", () => {
          if (Date.now() - dragStartTime < 500) {
            addToVisited(rd.node.id);
            navigateToSlug(rd.node.id);
          }
        });
    }

    // ---- Drag (canvas-level via D3) ---------------------------------

    let currentTransform = d3.zoomIdentity;

    d3.select(app.view).call(
      d3.drag()
        .container(() => app.view)
        .subject((event) => {
          const [mx, my] = d3.pointer(event, app.view);
          // Convert screen coords → simulation coords (centred at width/2, height/2)
          const sx = (mx - currentTransform.x) / currentTransform.k - width  / 2;
          const sy = (my - currentTransform.y) / currentTransform.k - height / 2;

          let closest = null, closestDist = Infinity;
          for (const n of simNodes) {
            const dx = (n.x || 0) - sx;
            const dy = (n.y || 0) - sy;
            const dist = dx * dx + dy * dy;
            const r = nodeRadius(n.id);
            if (dist < r * r * 9 && dist < closestDist) {
              closest = n; closestDist = dist;
            }
          }
          return closest;
        })
        .on("start", (event) => {
          if (!event.active) simulation.alphaTarget(1).restart();
          dragStartTime = Date.now();
          dragging = true;
          if (event.subject) {
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
            event.subject.__initDrag = { x: event.subject.x, y: event.subject.y };
          }
        })
        .on("drag", (event) => {
          if (!event.subject) return;
          const init = event.subject.__initDrag || { x: 0, y: 0 };
          event.subject.fx = init.x + (event.x - init.x) / currentTransform.k;
          event.subject.fy = init.y + (event.y - init.y) / currentTransform.k;
        })
        .on("end", (event) => {
          if (!event.active) simulation.alphaTarget(0);
          dragging = false;
          if (event.subject) { event.subject.fx = null; event.subject.fy = null; }
        })
    );

    // ---- Zoom + pan -------------------------------------------------

    d3.select(app.view).call(
      d3.zoom()
        .extent([[0, 0], [width, height]])
        .scaleExtent([0.25, 4])
        .on("zoom", ({ transform }) => {
          currentTransform = transform;
          stage.scale.set(transform.k, transform.k);
          stage.position.set(transform.x, transform.y);

          // Fade labels in with zoom
          const scaleOpacity = Math.max((transform.k - 1) / 3.75, 0);
          for (const rd of nodeRenderData) {
            if (hoveredId !== rd.node.id && rd.node.id !== focusSlug) {
              rd.label.alpha = scaleOpacity;
            }
          }
        })
    );

    // ---- Animation loop ---------------------------------------------

    let stopAnimation = false;

    function animate(time) {
      if (stopAnimation) return;

      TWEEN.update(time);

      // Update node + label positions
      for (const rd of nodeRenderData) {
        const { x, y } = rd.node;
        if (x == null || y == null) continue;
        const px = x + width  / 2;
        const py = y + height / 2;
        rd.gfx.position.set(px, py);
        rd.gfx.alpha = rd.alpha;
        rd.label.position.set(px, py);
      }

      // Redraw links each frame (positions change during simulation)
      for (let i = 0; i < linkRenderData.length; i++) {
        const ld  = linkRenderData[i];
        const lnk = simLinks[i];

        // After D3 resolves string ids to objects, source/target become node objects
        const src = typeof lnk.source === "object" ? lnk.source : null;
        const tgt = typeof lnk.target === "object" ? lnk.target : null;
        if (!src || !tgt) continue;

        ld.gfx.clear();
        ld.gfx.lineStyle(1, cssColorToNumber(ld.color), ld.alpha);
        ld.gfx.moveTo(src.x + width  / 2, src.y + height / 2);
        ld.gfx.lineTo(tgt.x + width  / 2, tgt.y + height / 2);
      }

      app.renderer.render(stage);
      requestAnimationFrame(animate);
    }

    requestAnimationFrame(animate);

    // ---- Resize handler ---------------------------------------------

    function onResize() {
      const newW = container.offsetWidth  || width;
      const newH = Math.max(container.offsetHeight || height, 200);
      app.renderer.resize(newW, newH);
    }
    window.addEventListener("resize", onResize);

    // ---- Cleanup ----------------------------------------------------

    return function cleanup() {
      stopAnimation = true;
      simulation.stop();
      window.removeEventListener("resize", onResize);
      try { app.destroy(true, { children: true }); } catch { /* ignore */ }
    };
  }

  // ------------------------------------------------------------------
  // Initialise all graph containers found on the page
  // ------------------------------------------------------------------

  const _cleanupFns = [];

  function _destroyAll() {
    for (const fn of _cleanupFns) {
      try { fn(); } catch { /* ignore */ }
    }
    _cleanupFns.length = 0;
  }

  async function initGraphContainers() {
    const localContainers  = document.querySelectorAll(".graph-container-local");
    const globalContainers = document.querySelectorAll(".graph-container-global");

    if (!localContainers.length && !globalContainers.length) return;

    // Determine where to fetch graph.json from
    const firstEl   = localContainers[0] || globalContainers[0];
    const graphUrl  = firstEl.dataset.graphUrl || "/graph.json";

    let graphData;
    try {
      const resp = await fetch(graphUrl);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      graphData = await resp.json();
    } catch (err) {
      console.error("[pelican_graph_view] Could not load", graphUrl, err);
      return;
    }

    const { nodes, links } = graphData;

    // Mark the current article as visited
    for (const el of localContainers) {
      if (el.dataset.slug) addToVisited(el.dataset.slug);
    }

    _destroyAll();

    // Render local graphs
    for (const el of localContainers) {
      const slug    = el.dataset.slug || null;
      const cfg     = _parseCfg(el.dataset.cfg);
      const canvas  = el.querySelector(".graph-canvas") || el;
      _cleanupFns.push(await renderGraph(canvas, slug, nodes, links, cfg));
    }

    // Render global graphs
    for (const el of globalContainers) {
      const cfg    = _parseCfg(el.dataset.cfg);
      const canvas = el.querySelector(".graph-canvas") || el;
      _cleanupFns.push(await renderGraph(canvas, null, nodes, links, cfg));
    }
  }

  function _parseCfg(raw) {
    try { return JSON.parse(raw || "{}"); } catch { return {}; }
  }

  // ------------------------------------------------------------------
  // Bootstrap
  // ------------------------------------------------------------------

  function bootstrap() {
    if (typeof d3     === "undefined") { console.error("[pelican_graph_view] D3 not loaded");      return; }
    if (typeof PIXI   === "undefined") { console.error("[pelican_graph_view] PixiJS not loaded");   return; }
    if (typeof TWEEN  === "undefined") { console.error("[pelican_graph_view] Tween.js not loaded"); return; }

    initGraphContainers().catch((err) =>
      console.error("[pelican_graph_view] Init error:", err)
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
