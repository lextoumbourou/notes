"""
pelican_graph_view — Obsidian-style interactive graph view for Pelican sites.

Adds:
- A local graph (per-article neighborhood) injected into article content
- A global graph page at /graph.html
- A graph.json data file at the output root
- Static assets (graph.js, graph.css) copied to {OUTPUT_PATH}/static/
"""

import json
import logging
import os
import re
import shutil
from copy import deepcopy

from pelican import signals

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "show_tags": False,
    "exclude_tags": [],
    "exclude_slugs": [],
    "include_hidden": False,
    "local": {
        "enabled": True,
        "depth": 1,
        "focus_on_hover": False,
    },
    "global": {
        "enabled": True,
        "slug": "graph",
        "title": "Graph View",
        "focus_on_hover": True,
        "enable_radial": True,
    },
    "repel_force": 0.5,
    "center_force": 0.3,
    "link_distance": 30,
    "font_size": 0.6,
    "scale_nodes_by_degree": True,
}

# Module-level store shared across signal handlers
_graph_state = {
    "nodes": [],
    "links": [],
    "cfg": {},
    "output_path": "",
}


def _merge_defaults(user_cfg):
    """Deep-merge user config over DEFAULTS, keeping nested dicts intact."""
    cfg = deepcopy(DEFAULTS)
    for key, value in user_cfg.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


# ---------------------------------------------------------------------------
# Signal: article_generator_init
# ---------------------------------------------------------------------------

def article_generator_init(generator):
    """Initialise per-generator storage."""
    generator._graph_slug_map = {}  # slug -> article
    generator._graph_edges = set()  # (source_slug, target_slug)


# ---------------------------------------------------------------------------
# Signal: article_generator_pretaxonomy
# ---------------------------------------------------------------------------

def article_generator_pretaxonomy(generator):
    """
    Build the full node + edge set from all articles.
    Runs after all articles have been read but before categories/tags are
    finalised, so all article objects are available.
    """
    cfg = _merge_defaults(generator.settings.get("GRAPH_VIEW", {}))
    _graph_state["cfg"] = cfg
    _graph_state["output_path"] = generator.settings.get("OUTPUT_PATH", "output")

    exclude_tags = set(cfg.get("exclude_tags", []))
    exclude_slugs = set(cfg.get("exclude_slugs", []))
    show_tags = cfg.get("show_tags", True)

    nodes = []
    links = []

    # Build slug → article map first so we can validate link targets
    include_hidden = cfg.get("include_hidden", False)
    all_articles = list(generator.articles)
    if include_hidden:
        all_articles += list(generator.hidden_articles)
    slug_map = {a.slug: a for a in all_articles}
    generator._graph_slug_map = slug_map

    for article in all_articles:
        slug = article.slug
        if slug in exclude_slugs:
            continue

        article_tags = []
        if hasattr(article, "tags") and article.tags:
            article_tags = [
                t.name for t in article.tags if t.name not in exclude_tags
            ]

        nodes.append(
            {
                "id": slug,
                "text": article.title,
                "tags": article_tags,
            }
        )

        # --- Outgoing links: scan rendered content for href="SLUG.html" ---
        if article._content:
            for match in re.finditer(r'href="([^"]+)\.html"', article._content):
                target_slug = match.group(1)
                # Strip any leading path components (e.g. "some/path/slug" → "slug")
                # The site uses flat slug URLs so just take the basename.
                target_slug = target_slug.split("/")[-1]
                if target_slug in slug_map and target_slug != slug:
                    links.append({"source": slug, "target": target_slug})

        # --- Tag nodes (optional) ---
        if show_tags:
            for tag_name in article_tags:
                tag_node_id = f"tag:{tag_name}"
                links.append({"source": slug, "target": tag_node_id})

    # Add tag pseudo-nodes if enabled
    if show_tags:
        tag_ids = {l["target"] for l in links if l["target"].startswith("tag:")}
        for tag_id in sorted(tag_ids):
            tag_name = tag_id[4:]  # strip "tag:" prefix
            nodes.append({"id": tag_id, "text": f"#{tag_name}", "tags": []})

    # Deduplicate links
    seen_links = set()
    deduped_links = []
    for lnk in links:
        key = (lnk["source"], lnk["target"])
        if key not in seen_links:
            seen_links.add(key)
            deduped_links.append(lnk)

    _graph_state["nodes"] = nodes
    _graph_state["links"] = deduped_links

    # Build inverted index: slug → list of (source_slug, source_title)
    backlink_map = {a.slug: [] for a in all_articles}
    for lnk in deduped_links:
        src, tgt = lnk["source"], lnk["target"]
        if tgt in backlink_map and src in slug_map:
            backlink_map[tgt].append({
                "slug": src,
                "title": slug_map[src].title,
            })
    _graph_state["backlink_map"] = backlink_map


# ---------------------------------------------------------------------------
# Signal: article_generator_write_article
# ---------------------------------------------------------------------------

def article_generator_write_article(generator, content):
    """Attach graph_html attribute to article for use in the template."""
    cfg = _graph_state.get("cfg") or _merge_defaults(
        generator.settings.get("GRAPH_VIEW", {})
    )
    local_cfg = cfg.get("local", DEFAULTS["local"])

    # Attach backlinks regardless of local graph enabled state
    backlink_map = _graph_state.get("backlink_map", {})
    content.graph_backlinks = backlink_map.get(content.slug, [])

    if not local_cfg.get("enabled", True):
        content.graph_html = ""
        return

    slug = content.slug

    js_cfg = json.dumps(
        {
            "depth": local_cfg.get("depth", 1),
            "focusOnHover": local_cfg.get("focus_on_hover", False),
            "showTags": cfg.get("show_tags", False),
            "repelForce": cfg.get("repel_force", 0.5),
            "centerForce": cfg.get("center_force", 0.3),
            "linkDistance": cfg.get("link_distance", 30),
            "fontSize": cfg.get("font_size", 0.6),
            "scaleNodesByDegree": cfg.get("scale_nodes_by_degree", True),
        },
        separators=(",", ":"),
    )

    content.graph_html = (
        f'<div class="graph-container-local"'
        f' data-slug="{slug}"'
        f' data-graph-url="/graph.json"'
        f" data-cfg='{js_cfg}'>\n"
        f'  <p class="graph-local-heading">Graph</p>\n'
        f'  <div class="graph-canvas-wrapper">\n'
        f'    <div class="graph-canvas"></div>\n'
        f'  </div>\n'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Signal: finalized
# ---------------------------------------------------------------------------

def finalized(pelican):
    """Write graph.json, graph.html, and copy static assets."""
    cfg = _graph_state.get("cfg") or _merge_defaults(
        pelican.settings.get("GRAPH_VIEW", {})
    )
    output_path = _graph_state.get("output_path") or pelican.settings.get(
        "OUTPUT_PATH", "output"
    )

    _write_graph_json(output_path)
    _copy_static_assets(output_path)

    global_cfg = cfg.get("global", DEFAULTS["global"])
    if global_cfg.get("enabled", True):
        _write_global_graph_html(output_path, cfg)


def _write_graph_json(output_path):
    graph_data = {
        "nodes": _graph_state["nodes"],
        "links": _graph_state["links"],
    }
    out_file = os.path.join(output_path, "graph.json")
    os.makedirs(output_path, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(graph_data, fh, ensure_ascii=False, indent=2)
    logger.info(
        "pelican_graph_view: wrote %s (%d nodes, %d links)",
        out_file,
        len(_graph_state["nodes"]),
        len(_graph_state["links"]),
    )


def _copy_static_assets(output_path):
    plugin_static = os.path.join(os.path.dirname(__file__), "static")
    dest_static = os.path.join(output_path, "static")
    os.makedirs(dest_static, exist_ok=True)
    for fname in ("graph.js", "graph.css"):
        src = os.path.join(plugin_static, fname)
        dst = os.path.join(dest_static, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.debug("pelican_graph_view: copied %s → %s", src, dst)
        else:
            logger.warning("pelican_graph_view: static asset not found: %s", src)


def _write_global_graph_html(output_path, cfg):
    global_cfg = cfg.get("global", DEFAULTS["global"])
    slug = global_cfg.get("slug", "graph")
    title = global_cfg.get("title", "Graph View")

    js_cfg = json.dumps(
        {
            "depth": -1,
            "focusOnHover": global_cfg.get("focus_on_hover", True),
            "enableRadial": global_cfg.get("enable_radial", True),
            "showTags": cfg.get("show_tags", True),
            "repelForce": cfg.get("repel_force", 0.5),
            "centerForce": cfg.get("center_force", 0.3),
            "linkDistance": cfg.get("link_distance", 30),
            "fontSize": cfg.get("font_size", 0.6),
            "scaleNodesByDegree": cfg.get("scale_nodes_by_degree", True),
        },
        separators=(",", ":"),
    )

    template_path = os.path.join(
        os.path.dirname(__file__), "templates", "graph_page.html"
    )
    with open(template_path, "r", encoding="utf-8") as fh:
        template = fh.read()

    html = (
        template.replace("{{ PAGE_TITLE }}", title)
        .replace("{{ GRAPH_CFG }}", js_cfg)
    )

    out_file = os.path.join(output_path, f"{slug}.html")
    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info("pelican_graph_view: wrote global graph page → %s", out_file)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register():
    signals.article_generator_init.connect(article_generator_init)
    signals.article_generator_pretaxonomy.connect(article_generator_pretaxonomy)
    signals.article_generator_write_article.connect(article_generator_write_article)
    signals.finalized.connect(finalized)
