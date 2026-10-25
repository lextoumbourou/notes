---
name: NotesByLex
description: Lex's digital garden as a hand-drawn technical zine. The frame is lettered and sketched, the reading column stays calm.
colors:
  paper: "#FBFAF6"
  paper-2: "#F2F0E8"
  ink: "#141414"
  ink-2: "#383836"
  muted: "#5C5B57"
  rule-soft: "#DCD9CF"
  highlighter-yellow: "#FFE84D"
  highlighter-pink: "#FF6FAE"
  highlighter-teal: "#2DD4BF"
  highlighter-violet: "#C4B2FF"
  highlighter-sky: "#A9D6FF"
  highlighter-orange: "#FFB866"
  highlighter-lime: "#BDEB7A"
  highlighter-tan: "#E2C59C"
  mark-ink: "#141414"
  link: "#0A6E63"
  link-hover: "#064C45"
  tag-bg: "#D5F2ED"
  tag-fg: "#0B4F48"
  slate: "#1B2023"
  slate-2: "#242A2E"
  chalk: "#EEEBE3"
  chalk-2: "#CFCBC2"
  chalk-muted: "#A9A79F"
  slate-rule: "#363D41"
  chalk-yellow: "#E9D54F"
  chalk-pink: "#F07BB0"
  chalk-teal: "#3FD3C0"
  chalk-violet: "#B7A2F5"
  chalk-sky: "#8FC3EE"
  chalk-orange: "#F0A35E"
  chalk-lime: "#A9D46F"
  chalk-tan: "#CDAF86"
  chalk-link: "#6FE0D0"
  chalk-link-hover: "#B2F3EA"
  slate-tag-bg: "#1F3A37"
  slate-tag-fg: "#B2EFE6"
typography:
  display:
    fontFamily: "Shantell Sans, Segoe Print, Bradley Hand, cursive"
    fontSize: "clamp(2.3rem, 1.5rem + 2.8vw, 3.6rem)"
    fontWeight: 700
    lineHeight: 1.1
  headline:
    fontFamily: "Shantell Sans, Segoe Print, Bradley Hand, cursive"
    fontSize: "clamp(1.9rem, 1.5rem + 1.4vw, 2.4rem)"
    fontWeight: 700
    lineHeight: 1.2
  title:
    fontFamily: "Shantell Sans, Segoe Print, Bradley Hand, cursive"
    fontSize: "1.75rem"
    fontWeight: 700
    lineHeight: 1.25
  side-heading:
    fontFamily: "Shantell Sans, Segoe Print, Bradley Hand, cursive"
    fontSize: "1.5rem"
    fontWeight: 700
    lineHeight: 1.15
  hand-label:
    fontFamily: "Shantell Sans, Segoe Print, Bradley Hand, cursive"
    fontSize: "1.05rem"
    fontWeight: 700
    lineHeight: 1.2
  body:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 400
    lineHeight: 1.65
  lede:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "1.35rem"
    fontWeight: 400
    lineHeight: 1.45
  meta:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.4
    fontFeature: "tnum"
  small:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
  code:
    fontFamily: "Atkinson Hyperlegible Mono, ui-monospace, SF Mono, Menlo, Consolas, monospace"
    fontSize: "0.92rem"
    fontWeight: 400
    lineHeight: 1.6
rounded:
  sketch-fallback: "3px"
  soft: "4px"
  pill: "999px"
  round: "50%"
spacing:
  gutter: "clamp(16px, 3vw, 40px)"
  measure: "42rem"
  sidebar: "360px"
  page-max: "1280px"
  block: "1.8em"
  section: "56px"
components:
  tag:
    backgroundColor: "{colors.tag-bg}"
    textColor: "{colors.tag-fg}"
    rounded: "{rounded.pill}"
    padding: "0 12px"
    height: "30px"
  tag-dark:
    backgroundColor: "{colors.slate-tag-bg}"
    textColor: "{colors.slate-tag-fg}"
    rounded: "{rounded.pill}"
    padding: "0 12px"
    height: "30px"
  nav-link:
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    padding: "4px 0"
  search-box:
    textColor: "{colors.ink}"
    rounded: "{rounded.soft}"
    padding: "8px 12px"
    width: "clamp(200px, 22vw, 300px)"
  code-block:
    textColor: "{colors.ink}"
    typography: "{typography.code}"
    rounded: "{rounded.sketch-fallback}"
    padding: "18px 20px"
  inline-code:
    backgroundColor: "{colors.paper-2}"
    textColor: "{colors.ink}"
    rounded: "{rounded.soft}"
    padding: "0.1em 0.38em"
  callout:
    textColor: "{colors.ink}"
    rounded: "{rounded.sketch-fallback}"
    padding: "16px 20px 4px"
  scribble:
    textColor: "{colors.link}"
    typography: "{typography.hand-label}"
    width: "min(15rem, 40%)"
  kind-paper:
    backgroundColor: "{colors.highlighter-yellow}"
    textColor: "{colors.mark-ink}"
    typography: "{typography.hand-label}"
    padding: "0 0.3em"
  kind-model:
    backgroundColor: "{colors.highlighter-pink}"
    textColor: "{colors.mark-ink}"
    typography: "{typography.hand-label}"
    padding: "0 0.3em"
  kind-essay:
    backgroundColor: "{colors.highlighter-teal}"
    textColor: "{colors.mark-ink}"
    typography: "{typography.hand-label}"
    padding: "0 0.3em"
  graph-node-current:
    backgroundColor: "{colors.highlighter-pink}"
    textColor: "{colors.mark-ink}"
    typography: "{typography.hand-label}"
  essay-row:
    textColor: "{colors.ink}"
    typography: "{typography.lede}"
    padding: "16px 0"
  essay-cover:
    rounded: "{rounded.sketch-fallback}"
    width: "132px"
  note-row:
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    padding: "12px 0"
  pagination-active:
    backgroundColor: "{colors.highlighter-yellow}"
    textColor: "{colors.mark-ink}"
    rounded: "{rounded.round}"
    size: "40px"
  theme-toggle:
    textColor: "{colors.ink}"
    rounded: "{rounded.round}"
    size: "42px"
---

# Design System: NotesByLex

## Overview

**Creative North Star: "The Napkin Zine"**

NotesByLex reads like a friend's napkin explanation bound into a technical zine. The frame is lettered and sketched: a marker wordmark, hand-drawn rules, wobbly rough.js outlines around code, search and graph nodes, and a single highlighter swipe under the page title. The reading column stays calm: an off-white page, a hyperlegible sans at a comfortable 42rem measure, and teal links underlined like any well-behaved text. Light mode is marker on paper; dark mode is chalk on slate.

The world refuses the three-column docs app shell and the cream-card blog. There are no cards inside cards and no drop shadows; structure comes from drawn lines, dashed rules and whitespace. Colour arrives as highlighter (yellow, pink, teal) laid over ink, never as filled panels. The hand appears in the frame and, only where Lex writes one, in a scribble margin note; the design never writes in his voice.

**Key Characteristics:**
- Shantell Sans lettering for every heading, the wordmark, graph labels, note kinds and scribbles; Atkinson Hyperlegible Next for everything you read at length.
- Sketched outlines (rough.js) with a plain CSS border or rule as the fallback, hidden once the sketch has drawn.
- One highlighter swipe per page, under the main title, drawn on once at load.
- Highlighter as the only fill colour: `==marks==`, note kinds, the current graph node, the active page number.
- Flat paper; depth comes from line weight, not shadow.

## Colors

Ink on warm off-white, with eight highlighters; dark mode swaps to chalk on slate and keeps the highlighters, slightly softened.

### Primary
- **Marker Ink** (ink): text, sketched outlines, the top bar rule, the sidebar rule, table header rules. In dark mode it becomes **Chalk** (chalk).
- **Deep Garden Teal** (link): links in prose and sidebars, focus ring, text cursor, the scribble's colour, callout icons. Hover darkens to **Forest Teal** (link-hover). Dark mode uses **Chalk Teal** (chalk-link) and lightens on hover (chalk-link-hover).

### Secondary
- **Highlighter Yellow** (highlighter-yellow): `==mark==` highlights, text selection, nav hover and current-page underlay, `paper` note kind, active page number, skip link, swipe on category and Topics pages. Dark: chalk-yellow.
- **Highlighter Pink** (highlighter-pink): the current note in the knowledge graph, `model` note kind, the current tag, swipe on tag pages. Dark: chalk-pink.
- **Highlighter Orange** (highlighter-orange): the `course` kind. Dark: chalk-orange.
- **Highlighter Lime** (highlighter-lime): the `talk` kind. Dark: chalk-lime.
- **Kraft Tan** (highlighter-tan): the `book` kind, muted like book cloth so it reads apart from course orange. Dark: chalk-tan.
- **Highlighter Sky** (highlighter-sky): the `note` kind, at 60% strength because notes are most rows. Dark: chalk-sky.
- **Highlighter Violet** (highlighter-violet): the `news` note kind only. Dark: chalk-violet.
- **Highlighter Teal** (highlighter-teal): title swipe on notes, pages, the home lede and All posts; link underline colour; list hover underlines; `essay` note kind. Dark: chalk-teal. Never used as text colour; text teal is the deepened link token.

### Neutral
- **Off-white Paper** (paper): page background. Dark: **Slate** (slate).
- **Second Sheet** (paper-2): inline code, notebook output, zebra rows in legacy tables. Dark: slate-2.
- **Soft Ink** (ink-2): secondary prose, blockquotes, table of contents, in-page citation links. Dark: chalk-2.
- **Pencil Grey** (muted): bylines, dates, counts, footer, placeholders. Dark: chalk-muted.
- **Faint Rule** (rule-soft): table row rules, keyboard hint border. Dark: slate-rule.
- **Mint Tag** (tag-bg / tag-fg): topic tags. Dark: slate-tag-bg / slate-tag-fg.
- **Mark Ink** (mark-ink): text on any highlighter fill; stays near-black in both themes so highlighted text keeps its contrast on chalk-mode highlighters.

### Named Rules
**The Highlighter Not Paint Rule.** Yellow, pink and teal are laid over ink as marks, underlays and swipes. They never fill a panel, card or band.

**The Two Teals Rule.** The bright highlighter teal decorates (swipes, underlines, fills); the deepened link teal carries text. Never set text in the highlighter.

**The Same Ink Rule.** Text on a highlighter is always mark ink, in both themes.

## Typography

**Display Font:** Shantell Sans (with Segoe Print, Bradley Hand, cursive), self-hosted variable 300 to 800
**Body Font:** Atkinson Hyperlegible Next (with system-ui), self-hosted variable 200 to 800 with italic
**Label/Mono Font:** Atkinson Hyperlegible Mono (with ui-monospace)

**Character:** A friendly marker hand on top, chosen by Lex from a specimen against the comp ("B · Shantell Sans"; "A marker font is fine"), over a sans built for legibility. The marker shouts the structure; the sans does the reading.

### Hierarchy
- **Display** (700, fluid 2.3rem to 3.6rem, 1.1): note and list page titles, balanced wrap. Home uses the headline size for its lede-as-title.
- **Headline** (700, fluid 1.9rem to 2.4rem, 1.2): prose H2, home H1 lede, Recent notes, Video and Comments headings.
- **Title** (700, 1.75rem, 1.25): prose H3, References heading, year headings on the All posts archive.
- **Side heading** (700, 1.5rem, 1.15): sidebar block headings such as On this page and Linked from.
- **Hand label** (600 to 700, about 1.05rem): note kinds, graph labels (15px, 600), scribbles (1.2rem, 600).
- **Body** (400, 1.125rem, 1.65): prose at a 42rem measure; paragraphs use pretty wrapping.
- **Lede** (400, 1.35rem, 1.45): home introduction. Essay titles use this size in sans bold (700, 1.3 line height). Articles show their written `summary` at this size in soft ink as a dek under the title, so titles can stay short; it is skipped when the note opens with the same text, or with `show_summary: false`. A paper note's "Notes on…" source line sits below the tags in small italic muted text above a dashed rule, so it never reads as the first paragraph.
- **Meta** (400, 0.94rem to 1rem, tabular numerals): bylines, dates, counts, list meta. H4 to H6 drop to the sans at 1.35rem bold.
- **Code** (400, 0.92rem, 1.6, mono): code blocks; inline code at 0.86em on the second sheet.

### Named Rules
**The Marker Frames, Sans Reads Rule.** Shantell Sans is for headings, the wordmark and short labels. Running prose, lists, tables and metadata are always Atkinson Hyperlegible Next.

**The Bought Hand Rule.** The hand is a marker font, not a personal handwriting font. Do not imitate Lex's handwriting.

## Layout

A single calm column with an optional sticky right sidebar. The page is capped at 1280px with a fluid gutter (16px to 40px); the top bar and footer align to the same edges. Prose and home content sit at a 42rem measure; list pages widen to 48rem. When a sidebar exists the grid becomes main plus a 360px column, separated by a hand-drawn vertical rule; the sidebar is sticky 24px from the top and scrolls on its own.

Rhythm is generous and vertical: 1.8em around code, callouts and blockquotes; 2.2em above H2; 56px before post-article sections; 36px between sidebar blocks. Lists of essays and notes are separated by dashed 1px rules, never boxes.

The site has two sections. **Essays** are the finished pieces and get the room: title, summary and date. **Notes** are working notes and stay compact: one row each, with the note's written summary in soft ink under its title and, when the note has a cover, a small 4:3 thumbnail on the right (112px, 80px on mobile). Home runs lede H1, bio, then one Latest feed of the 20 newest pieces of any kind, newest first, 56px below the bio: every entry uses the note row's grid (date, kind chip, title and summary, cover on the right), and essays and projects are sized up within it (bold title at essay size, summary at body size, a 132px cover, 96px on mobile), so finished pieces stand out by size while covers stay in one column. Its header links to All posts, and every kind chip in a feed row links to its kind page. Home's sidebar runs the Kinds key, then Topics, then Find me on. List pages (All posts, Essays, tag and category pages) carry a sidebar with a Kinds key (each kind as its highlighter chip with a count, linking to its page, current kind underlined) and the top 24 Topics (current topic bold with a pink underline); article pages keep their own sidebar and never show Topics. The All posts page (`notes.html`, URL kept for old links) is one year-grouped archive of every piece of any kind, newest first and not paginated, with essays and projects sized up as on home; the Essays page (`essays.html`) lists essays and projects together.

Responsive: at 1080px and below the sidebar drops under the article at the measure width with a top rule instead of the side rule. At 760px and below the nav wraps to its own row, search collapses to its icon until focused, note rows stack date and kind above the title, essay dates drop under the summary, and scribbles stop floating.

## Elevation & Depth

Flat. There are no box shadows anywhere in the system. Depth and grouping come from line: a 2px sketched rule under the top bar, 1.5px sketched outlines around code, callouts and search, a sketched vertical rule beside the sidebar, and dashed rules between list rows and above the footer. Hover states tint with a translucent highlighter rather than lifting.

### Named Rules
**The Drawn Line Rule.** Separate things with a drawn or dashed line, or with space. Never with a shadow, and never with a filled card.

**The Fallback Rule.** Every sketched element carries a plain CSS border or rule that matches its geometry; the sketch layer hides it only after drawing. The page must read correctly with JavaScript off.

## Shapes

Hand-drawn rough.js outlines are the primary form: rectangles with slight bowing (stroke 1.6, roughness 1.1), rules with more wobble (stroke 2, roughness 1.4), and a stable per-element seed so redraws never jitter. CSS fallbacks use a barely rounded corner (3px) for sketched boxes and 4px for search, images, embeds and inline code. Highlighter marks and note kinds use deliberately uneven corners (for example 0.6em 0.3em 0.8em 0.25em) so they read as a felt-tip stroke. Tags are pills; the theme toggle and pagination are circles. The "more" arrow is an authored hand-drawn SVG, inheriting the link colour.

## Components

### Top bar
- **Style:** wordmark left in Shantell Sans 700 (1.6rem to 2.1rem) with a black sketched underline (the logo), nav (All posts · Topics · About; `essays.html` still lists essays and projects, linked from the All posts intro rather than the nav), then sketched search and theme toggle at the right; a sketched 2px ink rule beneath.
- **Nav states:** ink text; hover and current page slide in a yellow highlighter underlay at 42% of the line height (220ms ease-out); current page is bold.
- **Mobile:** nav wraps to a full-width row; search collapses to its icon.

### Search
- **Style:** sketched box (1.5px ink fallback, 4px), transparent fill, search icon, mono keyboard hint.
- **Focus:** fills with 10% highlighter teal and hides the hint. Results drop into a paper panel with an ink border; hovered results take a 35% yellow underlay; matched words are highlighted yellow.

### Tags
- **Style:** mint pill (30px tall, 12px side padding, 0.94rem), smaller (26px) inside list meta.
- **State:** hover mixes in highlighter teal; the current tag mixes in highlighter pink.

### Links
- **Prose:** link teal, 1.5px teal underline offset 3px; hover darkens and adds an 18% teal underlay. In-page citations and footnotes are soft ink with a dotted underline.
- **Link lists** (table of contents, topics, note rows, list titles): no underline at rest, a 2px teal underline on hover. Socials and the tag cloud use a wavy teal underline on hover.

### Code and callouts
- **Code blocks:** transparent, sketched ink outline, 18px 20px padding, mono 0.92rem.
- **Callouts:** sketched outline, Shantell Sans title at 1.35rem with a 22px stroked icon in link teal (question callouts pink, warning and danger amber). No tinted backgrounds.

### Scribble (signature)
Lex's own margin note, rendered only where he writes `> [!scribble]`. Shantell Sans 600 at 1.2rem in link teal, floated right at min(15rem, 40%), rotated -3deg, with a drawn arrow above it. No outline, no title. Stops floating and rotates only -1.5deg on mobile.

### Highlight mark
`==text==` becomes a yellow highlighter stroke with a skewed gradient and uneven corners, cloned across line breaks, with mark ink text.

### Essay list
A contents page for finished pieces, the same on home and the Essays page. Each entry is a bold sans title (1.35rem, 700, balanced wrap) in ink, its summary as a dek in soft ink (line height 1.5) below, and a month-year date (meta, tabular) right-aligned on the title's baseline; 16px vertical padding and a dashed rule between entries. An essay with a cover image leads with a 132px 4:3 thumbnail (cropped to fill, 3px corners) spanning the rows on the left, title and dek pinned to the top of the middle column, date on the right. An essay without a cover keeps the two-column text row. Title hover adds the 2px teal underline. On mobile the thumbnail shrinks to 88px and the date drops under the dek.

### Note rows
Compact one-line entries for working notes: date, kind, title, on a three-column grid (7.5rem, 4.5rem, the rest) with 12px padding and dashed rules. Titles are body sans at 1.1rem in ink, teal underline on hover. In the year-grouped archive the date column narrows to 4.5rem and drops the year, under Shantell Sans year headings (title size) with a mono count in pencil grey.

### Note kinds
Category labels in note rows and list meta, lettered in Shantell Sans 700. The kinds are `essay` and `project` (finished pieces, Essays section) and `note`, `paper`, `model`, `course`, `book`, `talk` and `news` (working notes). Papers get a yellow highlighter, models pink, news violet, notes a lighter sky blue (they are most rows), courses orange, talks lime, books kraft tan, essays and projects teal. Every kind now has a colour. In dark mode kinds use the solid chalk highlighters (blending toward transparent darkens them on slate and costs contrast). A category page hides the kind on its own entries, since the page title already says it.

### Sketched knowledge graph (signature)
A sidebar graph of the note's neighbours drawn with rough.js: notes linking in sit above the current note, notes it links out to sit below, in rows that fit the column; a dense side switches to pairs either side of a central trunk. Nodes are soft sketched boxes filled with paper and labelled in Shantell Sans; the current note is filled highlighter pink with a 2px stroke. Edges are curved doodle arrows showing link direction. Hover turns a node's outline and label link teal. Overflow becomes a "more connections" link with the hand-drawn arrow.

### Title swipe (signature)
One highlighter stroke (7px, bowed) under the last line of the page's main title, as wide as that line, drawn on once at load (700ms, ease-out, 150ms delay) and static under reduced motion. Teal on notes, pages, home and All posts; yellow on category and Topics pages; pink on tag pages.

### Diagrams
Cut-paper blocks: flat highlighter-filled rectangles with no stroke, rotated 1 to 2 degrees, sized to their Shantell Sans 800 labels (48px horizontal and 30px vertical padding at 1920 by 1080), joined by straight ink arrows 20px thick with solid triangular heads. Arrow labels in Shantell Sans 700. Kind colours carry over (LLM or model steps in highlighter pink); at most five fills per diagram; mark ink on every fill; paper ground. Reference: `notes/_media/agent-harness/agent-harness-loop-colour.png`.

For tensors and maths, the same flat paper, drawn to match the code:
- **Vectors** are a row of flat square cells in one highlighter, with an ellipsis cell standing in for the hidden dimensions. Never draw every dimension.
- **Matrices follow the code's layout.** In PyTorch-style code each token is a row, so stack one vector per token, labelled with the token in Shantell Sans 700 beside it. Do not switch to textbook column vectors when the article's code uses rows.
- **One colour per tensor, kept across a series** (in the self-attention set: X lime, query yellow, key orange, value teal, output pink). The colour follows the tensor from diagram to diagram.
- **Score and weight tables** are flat second-sheet cells with Atkinson Hyperlegible Mono numbers, row and column labels as tokens, and the axis names (query, key) in pencil grey. Show magnitude with one highlighter mixed in proportion to the value, never a multi-colour heat map. Highlight a single cell only when the prose points at it.
- **The batch is drawn gently**: a second sheet just behind the tensor, labelled "batch" in pencil grey, with the full shape in mono below, for example `(1, 4, 1024)`.
- **Sums** gather their inputs with plain ink lines (no heads) into a Shantell Sans "+", then one arrow to the result. Arrowheads that meet at one point pile up.
- **Captions** sit at the bottom in Shantell Sans 700, one plain sentence; code goes beneath in mono, set exactly as it appears in the article.
- **Legibility on a phone**: nothing below 30px at 1920 by 1080, and numbers at 46px or more, because a phone shows the image at about a fifth of its size.
- **A step-by-step series carries one worked example throughout** with consistent numbers, so the scores in one diagram produce the weights in the next. Reference: `notes/_media/self-attention/`.

## Do's and Don'ts

### Do:
- **Do** put exactly one highlighter swipe on a page, under the main title only, drawn on once at load.
- **Do** keep links in running prose underlined; underline link lists on hover only.
- **Do** keep the wordmark's black sketched underline; it is the logo.
- **Do** make the home lede the H1 and keep the nav as All posts · Topics · About, with the wordmark as the home link.
- **Do** give essays the room (cover thumbnail when the essay has one, title, summary dek, month-year date) and keep notes as compact rows, with at most a small cover thumbnail on the right; on home and the All posts page, every kind shares one newest-first feed on the note row's grid, essays sized up.
- **Do** give every sketched element a CSS border or rule fallback of the same geometry.
- **Do** set text on highlighter fills in mark ink (#141414) in both themes.
- **Do** keep light mode as marker on paper and dark mode as chalk on slate, and meet WCAG 2.2 AA in both.

### Don't:
- **Don't** split the All posts archive by kind or paginate it; it is one year-grouped page, filtered through kind pages.
- **Don't** put swipes on section, sidebar or list headings ("they're a bit busy being on every title").
- **Don't** add read-tracking marks such as visited ticks or progress ("My viewers don't need to track that").
- **Don't** write copy in Lex's voice; scribbles appear only where Lex writes `> [!scribble]`.
- **Don't** use em dashes in any prose the design adds.
- **Don't** use box shadows, cards inside cards, or highlighter-filled panels.
- **Don't** set running text in Shantell Sans or in the bright highlighter teal.
- **Don't** use a handwriting font that imitates Lex's own hand.
