"""Print a quick summary of a CSV file. Standard library only."""
import argparse
import csv
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--by", help="column to group by")
    parser.add_argument("--sum", help="numeric column to total per group")
    args = parser.parse_args()

    with open(args.path, newline="") as f:
        rows = list(csv.DictReader(f))

    columns = list(rows[0]) if rows else []
    print(f"rows: {len(rows)}")
    print(f"columns: {', '.join(columns)}")

    if args.by:
        totals = Counter()
        for row in rows:
            totals[row[args.by]] += float(row[args.sum]) if args.sum else 1
        label = f"total {args.sum}" if args.sum else "rows"
        print(f"\n{label} by {args.by}:")
        for key, value in totals.most_common():
            print(f"  {key}: {value:g}")


if __name__ == "__main__":
    main()
