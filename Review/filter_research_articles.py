#!/usr/bin/env python3
"""
Keep only research-article records from review outputs.

Rules:
- Scopus row is kept only if subtype_description == "Article".
- WoS row is kept only if types contains "Article" and does NOT contain
  conference/review/editorial style markers.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


WOS_EXCLUDE_TOKENS = {
    "conference paper",
    "proceedings paper",
    "review",
    "editorial material",
    "letter",
    "meeting abstract",
    "book review",
}


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def is_scopus_research_article(row: dict[str, str]) -> bool:
    subtype = normalize(row.get("subtype_description", ""))
    return subtype == "article"


def is_wos_research_article(row: dict[str, str]) -> bool:
    types_raw = normalize(row.get("types", ""))
    if not types_raw:
        return False

    type_items = [normalize(x) for x in types_raw.split(";") if normalize(x)]
    has_article = any(t == "article" for t in type_items)
    has_excluded = any(t in WOS_EXCLUDE_TOKENS for t in type_items)
    return has_article and (not has_excluded)


def is_research_article(row: dict[str, str]) -> bool:
    source_db = normalize(row.get("source_db", ""))
    if source_db == "scopus":
        return is_scopus_research_article(row)
    if source_db == "webofscience":
        return is_wos_research_article(row)

    if "subtype_description" in row:
        return is_scopus_research_article(row)
    if "types" in row:
        return is_wos_research_article(row)
    return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter research articles only")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_csv(in_path)
    filtered = [r for r in rows if is_research_article(r)]
    write_csv(out_path, filtered)

    print(f"Input   : {in_path} ({len(rows)})")
    print(f"Output  : {out_path} ({len(filtered)})")
    print(f"Removed : {len(rows) - len(filtered)}")


if __name__ == "__main__":
    main()
