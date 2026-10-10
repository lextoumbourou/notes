# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary reader: Lex himself. NotesByLex is his public second brain, a Zettelkasten he writes in, returns to and wanders through. Other readers are welcome but secondary: curious, technically minded people (CS students through to principal engineers) who usually arrive on a single note from search, a social post or a YouTube description, then may follow links outward.

## Product Purpose

A digital garden of working notes on AI, machine learning, software engineering, learning and assorted topics, collected since 2012 and kept with the Zettelkasten method since 2020. Notes are continually revised rather than finished. Success is a place Lex wants to write in and wander through, where following a link from one idea to the next is a pleasure, and where any reader landing on one note can understand it and find its neighbours.

## Positioning

What Lex actually teaches is how to learn: he learns in public by teaching concepts as he learns them ("the best way to learn is to teach"). The site is a living, interlinked knowledge graph rather than a feed of finished posts. A neighbouring blog cannot truthfully copy 1,000+ permanent notes connected by real backlinks, paper notes written to a consistent template, and a graph that shows how ideas connect.

## Operating Context

- Notes are written in Obsidian in Lex's private vault, then published into this Pelican site and deployed to notesbylex.com via Netlify.
- Content kinds (Pelican categories). Finished pieces, listed under Essays: `essay` (pieces Lex would post on Hacker News) and `project` (write-ups of things he built). Working notes, listed under Notes: `note` (idea notes, the bulk; the default when a note has no category), `paper` (paper summaries), `model` (model and model-family notes), `course`, `book`, `talk` (notes from courses, books, and talks or videos) and `news`.
- Articles are the source for NotesByLex YouTube videos; posts are shared to Bluesky, Mastodon, Threads and LinkedIn, and replies there appear on the article as comments.
- Readers arrive mid-graph far more often than at the home page.

## Capabilities and Constraints

- Static site: Pelican with Jinja templates in `theme/`, custom Markdown extensions (Obsidian callouts, Mermaid, notebook fences, ToC headings), KaTeX maths, Pygments code highlighting, Pagefind search, pelican-cite citations with a generated References section, image lightbox.
- Per-note knowledge graph (`pelican_graph_view`, D3 + Pixi) and "Linked from" backlinks: must remain visible features.
- Light and dark themes with a manual toggle: must remain.
- Bluesky and Mastodon comment threads, YouTube embeds, cover images with credits, paper-note ledes.
- URLs (`{slug}.html`), feeds and existing content are fixed.
- Drafts are hidden from the site but the repository is a public GitHub repo: nothing private belongs in templates or this file.
- Accessibility baseline: WCAG 2.2 AA contrast and keyboard access, in both themes.

## Brand Commitments

- Name: **NotesByLex** is the house (the blog and the NotesByLex YouTube channel); Lex Toumbourou is clearly the author behind it. Keep the house distinct from Uncle Lex, his separate creative alias.
- Voice: the prose is Lex's own. Design and tooling never generate or reword copy in his voice. Plain, curious, first-person, unpretentious; he loves learning and it shows.
- Writing style: never use em dashes.
- Existing brand feel stated for the video channel: intellectual and clean, slightly warm, consistent.
- Open decision: the current Canva Sans typeface is Canva's proprietary brand font and is not a brand commitment; replacing it is expected.

## Evidence on Hand

- 1,000+ permanent notes, ~60 paper notes, model notes, essays and course notes in `notes/`.
- Real backlink graph data and citations (`notes/citations.bib`).
- Cover images with photo credits in `notes/_media/`.
- No portrait, logo or illustration system exists yet; none should be fabricated.

## Product Principles

1. The garden is for wandering: every note should make its neighbours easy and tempting to reach.
2. Reading comes first: a reader landing on one note sees that note immediately, on any device.
3. Living, not finished: dates, revision and connection matter more than publication polish.
4. Learning in public: show the scaffolding (sources, links, references) rather than hiding it.
5. Lex's words, unedited: the design frames the writing and never speaks for him.

## Accessibility & Inclusion

WCAG 2.2 AA in light and dark themes; respect `prefers-reduced-motion` (the graph animates); full keyboard access for search and theme toggle.
