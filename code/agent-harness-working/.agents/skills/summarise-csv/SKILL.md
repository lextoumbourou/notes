---
name: summarise-csv
description: Summarise a CSV file (row count, columns, totals and the biggest groups). Use when the user asks what is in a CSV, wants quick stats, or asks for a breakdown of a CSV by one of its columns.
---

# Summarise a CSV

1. Run the bundled script `scripts/summarise.py` (relative to this skill's folder) on the file:

   ```bash
   python3 scripts/summarise.py data/<file>.csv
   ```

   To break a numeric column down by another column, add `--by <column> --sum <column>`:

   ```bash
   python3 scripts/summarise.py data/notesbylex-notes-by-year.csv --by kind --sum notes
   ```

2. Report what the script prints in three to five short bullets: what one row represents, the size of the file, and the one or two most interesting numbers.
3. Do not guess at columns the script did not show. If the user wants a chart, say which columns you would plot instead of drawing one.
