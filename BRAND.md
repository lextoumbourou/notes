# NotesByLex Brand Guide

NotesByLex is Lex Toumbourou's public second brain: a digital garden of working notes and finished essays on AI, software, learning and assorted topics, kept since 2012 with the Zettelkasten method. It reads like a friend's napkin explanation bound into a technical zine. The frame is lettered and sketched; the reading column stays calm.

This brand book is the readable version of `PRODUCT.md` (who it is for, purpose, voice) and `DESIGN.md` (the visual system), which sit beside it.

## The brand in one breath

- **House and author.** NotesByLex is the house, for the blog and the NotesByLex YouTube channel. Lex Toumbourou is clearly the author behind it. Keep it separate from Uncle Lex, his creative alias.
- **Who it is for.** Lex first: he writes in it and wanders through it. Then curious, technically minded readers who land on one note from search, a social post or a video description.
- **What it stands for.** Learning in public by teaching. The site shows its scaffolding: sources, links, references, revisions.
- **Two kinds of writing.** **Essays** are finished pieces Lex would be proud to post on Hacker News, plus **projects** (write-ups of things he built). **Notes** are working notes that help him learn: idea notes, papers, models, courses, books, talks and news. Essays get room; notes stay compact.

## Content fundamentals

- **Lex's words, unedited.** All prose is Lex's own. Never write, reword or "improve" copy in his voice, and never invent quotes, scribbles or claims. Interface labels (Notes, Topics, All essays) are the only words the design supplies, and they stay plain and third person.
- **Voice.** Plain, curious, first person, unpretentious, honest about uncertainty. Real examples from the site: "These are my working notes that I've been collecting over the last decade+ using the Zettelkasten method." and "I don't know how useful the notes' contents will be to others. However, I've made them public because it helps me motivate myself to create these notes, and you never know."
- **Punctuation.** Never use em dashes. Use commas, parentheses, colons or separate sentences.
- **Casing.** Sentence case for interface labels and headings the design adds. Note titles keep the casing Lex gave them.
- **No emoji** in the interface. Social links use real brand icons, not emoji.
- **Tags** are CamelCase topic names (MachineLearning, LargeLanguageModels), shown as written.

## Name and wordmark

- There is no logo file. The wordmark is the name **NotesByLex** set in `hand` (Shantell Sans 700) with a black sketched underline in `ink`. The underline is part of the mark; keep it.
- Set the name as one word, capital N, B and L. In running text "NotesByLex" or "notesbylex.com".
- The wordmark is the home link. Do not repeat the name as a page title on the home page; the home H1 is the lede.

## Colour

Light is **marker on paper**, dark is **chalk on slate**. Define every colour from the tokens; both themes ship together.

- Set the page on `paper`, text in `ink`, secondary prose in `ink-2`, dates and counts in `muted`.
- Colour arrives as **highlighter**, laid over ink: `hl-yellow`, `hl-pink`, `hl-teal`, `hl-violet` for news, a lighter `hl-sky` for notes, `hl-orange` for courses, `hl-lime` for talks and a muted `hl-tan` for books. Use them for marks, underlays, swipes and note kinds. Never fill a panel, card or band with them.
- **Two teals.** `hl-teal` decorates (swipes, underlines, the essay kind). `link` carries text (links, scribbles, callout icons). Never set text in `hl-teal`.
- **Same ink.** Text on any highlighter fill is always `mark-ink`, in both themes.
- Topic tags use `tag-bg` with `tag-fg`. Inline code sits on `code-bg`.
- The focus ring is `focus`, 2.5px solid with a 3px offset.
- Every text pair meets WCAG 2.2 AA in both themes: `ink` and `muted` on `paper`, `link` on `paper`, `tag-fg` on `tag-bg`, `mark-ink` on each highlighter.

## Typography

- **Marker frames, sans reads.** Use `hand` (Shantell Sans) only for headings, the wordmark, note kinds, graph labels and scribbles. Set all running text, lists, tables and metadata in `sans` (Atkinson Hyperlegible Next). Code is `mono` (Atkinson Hyperlegible Mono).
- Page titles use `display`; prose H2 and home section headings use `headline`; H3, References and year headings use `title`; sidebar headings use `side-heading`; kinds use `hand-label`.
- Body copy is `body` at the `measure` (42rem, about 70 characters). Bylines and dates use `meta` with tabular numerals.
- The hand is a bought marker font. Never use a font that imitates Lex's own handwriting.
- All three faces are self-hosted (SIL Open Font License). Canva Sans is not part of this brand.

## Lines, shapes and depth

- **Drawn line, not shadow.** Separate things with a hand-drawn line (rough.js), a dashed rule, or space. No box shadows, no cards inside cards.
- Sketched outlines wrap code blocks, callouts, search and graph nodes; a sketched rule runs under the top bar and beside the sidebar. Every sketch has a CSS border fallback of the same shape (`sketch-fallback` 3px corner), so the page reads with JavaScript off.
- Highlighter shapes have deliberately uneven corners, like a felt-tip stroke. Tags are pills (`pill`); the theme toggle and active page are circles (`round`).

## Signature moves

- **One swipe per page.** A single highlighter stroke under the last line of the main title, drawn on once at load (700ms ease-out) and static under reduced motion. Teal on notes and pages, yellow on category and Topics pages, pink on tag pages. Never on section, sidebar or list headings.
- **Scribbles.** Lex's own margin notes, rendered only where he writes `> [!scribble]`: `hand` at 1.2rem in `link`, rotated slightly, with a drawn arrow. Never generate one.
- **Sketched knowledge graph.** The sidebar draws the note's neighbours from real link data: notes linking in above, links out below, curved doodle arrows, the current note filled `hl-pink`.
- **Highlights.** Obsidian's `==text==` renders as a `hl-yellow` felt-tip mark.

## Diagrams

- **Colourful but simple.** Draw concepts as cut-paper blocks: flat rectangles filled with a highlighter colour, no outlines, each turned a degree or two, sized to their words with even padding.
- **Chunky ink arrows.** Thick straight `ink` arrows with solid heads join the blocks. Small `hand` labels sit beside an arrow when it needs a name (for example "tool call").
- **Colour means something.** Reuse the kind colours where they fit (an LLM or model step is `hl-pink`), and otherwise give each step its own highlighter so the loop reads at a glance. No more than five colours in one diagram.
- **Type.** Block labels in `hand` (Shantell Sans 800), dark `mark-ink` on every fill; code in `mono`.
- **Ground.** Diagrams sit on `paper`, so they read as a sheet in either theme.
- **Words.** Labels quote the article; never invent them.
- The agent loop in the Agent Harness post is the reference example; its editable source lives in the vault.

## Links

- Links in running prose are always underlined (1.5px, `hl-teal` underline, `link` text).
- Link lists (table of contents, Linked from, topics, note rows, essay titles) underline on hover only.
- In-page citations and footnotes are `ink-2` with a dotted underline.

## Layout

- One calm column at the `measure`, with an optional sticky `sidebar` (360px) beside it, inside `page-max` (1280px) and a fluid `gutter`.
- Home: the lede as H1, a short bio, **Essays** (cover thumbnail, title, summary, month and year, matching the Essays page), then **Notes** (12 compact rows: date, kind, title, no images).
- Navigation: Essays · Notes · Topics · About, with the wordmark as home.
- List pages (Notes, Essays, tags, categories) have a sidebar: a Kinds key that doubles as the colour legend, then Topics. Articles keep On this page, Linked from and the graph instead.
- The Notes page is one year-grouped archive of every working note. The Essays page lists essays and projects together, with no kind chips.
- Note kinds: `note` (`hl-sky`, lighter), `paper` (`hl-yellow`), `model` (`hl-pink`), `news` (`hl-violet`), `course` (`hl-orange`), `talk` (`hl-lime`), `book` (`hl-tan`); finished kinds `essay` and `project` (`hl-teal`).
- Below 1080px the sidebar moves under the article; below 760px the nav wraps and rows stack.

## Iconography

- **Interface icons:** Lucide (`file-text` beside backlinks; callout icons), stroke style, inheriting colour.
- **Social icons:** Simple Icons glyphs for Bluesky, Mastodon, LinkedIn, GitHub and RSS, filled with `ink`.
- **The "more" arrow:** one authored hand-drawn SVG arrow after "All essays", "All notes" and similar links, in `link`. Never a text glyph such as →.
- No emoji, no decorative icon tiles.

## Motion

- One authored moment: the title swipe drawing on. Everything else is a quick hover tint (about 160 to 220ms, ease-out). Respect `prefers-reduced-motion` everywhere.

## Do and don't

- **Do** keep essays first on home with summaries, and notes as compact rows.
- **Do** keep every sketch backed by a CSS fallback.
- **Do** quote Lex's real words when you need an example of the voice.
- **Don't** add read-tracking marks (visited ticks, progress bars).
- **Don't** put swipes on more than the page title.
- **Don't** write copy in Lex's voice or use em dashes.
- **Don't** use shadows, filled highlighter panels, or running text in the marker face.

## Where the details live

- `PRODUCT.md`: who the site is for, its purpose, voice and constraints.
- `DESIGN.md`: every token, type style and component rule, derived from the shipped theme.
- `theme/static/css/style.css` and `theme/static/js/sketch.js`: the implementation.
