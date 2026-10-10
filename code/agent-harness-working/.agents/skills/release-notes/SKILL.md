---
name: release-notes
description: Write short release notes from a git history. Use when the user asks for release notes, a changelog entry, or a summary of what changed between two git refs or over a period of time.
---

# Release notes from git

1. Find the range. If the user names two refs, use them; otherwise use the last 20 commits:

   ```bash
   git log --no-merges --pretty=format:'%h %s' -20
   ```

2. Group the commits under at most three headings: **New**, **Fixed** and **Changed**. Drop commits that only touch formatting, typos or the build.
3. Write one line per change in plain English, starting with a verb ("Add", "Fix", "Remove"). Put the short hash in brackets at the end.
4. Keep the whole thing under 15 lines and never invent a change that is not in the log.
