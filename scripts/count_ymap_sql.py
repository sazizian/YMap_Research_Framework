"""Count extracted Y-Map SQL files and compare them with candidates.csv."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys


EXPECTED = 12_500
DEFAULT_ROOT = Path("runs/ymap_cohort_12500_sql")
SQL_SUBDIR = Path("Dataset/YMap/All_SQL_Solutions")


def show_paths(title: str, paths: set[str]) -> None:
    print(f"{title}: {len(paths):,}")
    for path in sorted(paths)[:10]:
        print(f"  {path}")
    if len(paths) > 10:
        print(f"  ... and {len(paths) - 10:,} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, nargs="?", default=DEFAULT_ROOT,
        help="extraction folder containing candidates.csv and Dataset (default: %(default)s)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    sql_dir = root / SQL_SUBDIR
    if not sql_dir.is_dir():
        print(f"SQL folder not found: {sql_dir}", file=sys.stderr)
        print("Pass the folder where you extracted the ZIP.", file=sys.stderr)
        return 2

    actual = {
        path.relative_to(root).as_posix()
        for path in sql_dir.rglob("*")
        if path.suffix.lower() == ".sql" and path.is_file()
    }
    print(f"Folder: {root}")
    print(f"SQL files found: {len(actual):,} (expected: {EXPECTED:,})")
    manifest = root / "candidates.csv"
    if not manifest.is_file():
        print("candidates.csv is missing. The count is shown, but paths cannot be verified.")
        return 2 if len(actual) == EXPECTED else 1

    with manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"sql_path", "scenario"} <= set(reader.fieldnames or []):
            print("candidates.csv is missing sql_path or scenario.", file=sys.stderr)
            return 2
        rows = list(reader)
    counts = Counter(row["sql_path"] for row in rows)
    if any(not path for path in counts):
        print("candidates.csv contains an empty SQL path.", file=sys.stderr)
        return 2

    listed = set(counts)
    duplicates = {path for path, count in counts.items() if count > 1}
    matched_by_scenario = Counter(row["scenario"] for row in rows if row["sql_path"] in actual)
    print(f"Candidates listed: {len(rows):,}")
    print(f"Native SQL files matched: {matched_by_scenario['native']:,}")
    print(f"Archive SQL files matched: {matched_by_scenario['archive']:,}")
    missing, extra = listed - actual, actual - listed
    show_paths("Missing", missing)
    show_paths("Unexpected", extra)
    if duplicates:
        show_paths("Duplicate paths in candidates.csv", duplicates)

    if len(actual) == len(rows) == EXPECTED and not (missing or extra or duplicates):
        print("PASS: All 12,500 SQL files match the list.")
        return 0
    print("FAIL: The extracted SQL files do not match the expected list.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Could not check the SQL files: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
