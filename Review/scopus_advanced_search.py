#!/usr/bin/env python3
"""
Scopus advanced-search metadata collector for research-article positioning.

Usage example:
  set ELS_API_KEY=YOUR_KEY
  python Review/scopus_advanced_search.py --max-results-per-query 300 --view COMPLETE

Optional:
  set ELS_INSTTOKEN=YOUR_INSTTOKEN
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"


@dataclass(frozen=True)
class QuerySpec:
    qid: str
    description: str
    query: str


DEFAULT_QUERIES: list[QuerySpec] = [
    QuerySpec(
        qid="Q1_hems_der_tariffs",
        description="HEMS with PV+BESS+EV under dynamic tariffs",
        query='TITLE-ABS-KEY(("home energy management" OR HEMS OR "residential energy management") AND (PV OR photovoltaic) AND (battery OR BESS) AND (EV OR "electric vehicle") AND ("dynamic tariff" OR "time-of-use" OR "real-time pricing"))',
    ),
    QuerySpec(
        qid="Q2_milp_mpc_online",
        description="Classical optimization baselines",
        query='TITLE-ABS-KEY((HEMS OR "residential energy management") AND (MILP OR "mixed-integer linear programming" OR MPC OR "model predictive control") AND ("real-time" OR online))',
    ),
    QuerySpec(
        qid="Q3_rl_energy",
        description="RL for energy management",
        query='TITLE-ABS-KEY(("reinforcement learning" OR "deep reinforcement learning") AND ("home energy management" OR microgrid) AND (PV OR battery OR EV))',
    ),
    QuerySpec(
        qid="Q4_offpolicy_vs_onpolicy",
        description="SAC and other RL families",
        query='TITLE-ABS-KEY(("soft actor-critic" OR SAC OR TD3 OR DDPG OR PPO) AND ("energy management" OR microgrid OR HEMS))',
    ),
    QuerySpec(
        qid="Q5_safe_rl_constraints",
        description="Safe/Constrained RL and action projection",
        query='TITLE-ABS-KEY(("safe reinforcement learning" OR "constrained reinforcement learning" OR "action projection" OR "safety layer") AND ("energy management" OR microgrid OR "power systems"))',
    ),
    QuerySpec(
        qid="Q6_pomdp_temporal_models",
        description="Partial observability and temporal architectures",
        query='TITLE-ABS-KEY((POMDP OR "partial observability") AND ("reinforcement learning") AND (GRU OR LSTM OR TCN OR Transformer OR Attention))',
    ),
    QuerySpec(
        qid="Q7_il_plus_rl",
        description="Imitation learning + RL hybrid pipelines",
        query='TITLE-ABS-KEY(("imitation learning" OR "behavior cloning") AND ("reinforcement learning") AND ("warm start" OR pretraining OR initialization) AND (energy OR microgrid OR HEMS))',
    ),
    QuerySpec(
        qid="Q8_multi_tariff_generalization",
        description="Tariff-aware robustness/generalization",
        query='TITLE-ABS-KEY(("dynamic pricing" OR "time-of-use" OR tariff) AND ("energy management") AND (robustness OR generalization OR "multi-tariff"))',
    ),
]


def build_filtered_query(
    base_query: str,
    min_year: int,
    language: str,
    doctype_codes: list[str],
) -> str:
    doctype_clause = " OR ".join(f"DOCTYPE({code})" for code in doctype_codes)
    return (
        f"{base_query} AND PUBYEAR > {min_year - 1} "
        f"AND LANGUAGE({language}) AND ({doctype_clause})"
    )


def http_get_json(
    url: str,
    params: dict[str, Any],
    api_key: str,
    insttoken: str | None,
    retries: int = 5,
    timeout_sec: int = 60,
) -> dict[str, Any]:
    query_string = urlencode(params)
    full_url = f"{url}?{query_string}"

    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": api_key,
    }
    if insttoken:
        headers["X-ELS-Insttoken"] = insttoken

    for attempt in range(retries):
        req = Request(full_url, headers=headers, method="GET")
        try:
            with urlopen(req, timeout=timeout_sec) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload)
        except HTTPError as exc:
            status = exc.code
            if status in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {status}: {body[:800]}") from exc
        except URLError as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Network error: {exc}") from exc

    raise RuntimeError("Unreachable retry state")


def parse_authors(entry: dict[str, Any]) -> str:
    authors = entry.get("author", [])
    names = [a.get("authname", "").strip() for a in authors if a.get("authname")]
    return "; ".join(names)


def parse_affiliations(entry: dict[str, Any]) -> str:
    affiliations = entry.get("affiliation", [])
    names = [a.get("affilname", "").strip() for a in affiliations if a.get("affilname")]
    return "; ".join(names)


def parse_scopus_link(entry: dict[str, Any]) -> str:
    for link in entry.get("link", []):
        if link.get("@ref") == "scopus":
            return link.get("@href", "")
    return ""


def parse_entry(query_id: str, query_text: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "query": query_text,
        "eid": entry.get("eid", ""),
        "title": entry.get("dc:title", ""),
        "doi": entry.get("prism:doi", ""),
        "publication_name": entry.get("prism:publicationName", ""),
        "cover_date": entry.get("prism:coverDate", ""),
        "subtype_description": entry.get("subtypeDescription", ""),
        "citedby_count": entry.get("citedby-count", ""),
        "openaccess": entry.get("openaccess", ""),
        "authors": parse_authors(entry),
        "affiliations": parse_affiliations(entry),
        "scopus_link": parse_scopus_link(entry),
    }


def search_scopus_query(
    query_spec: QuerySpec,
    api_key: str,
    insttoken: str | None,
    min_year: int,
    language: str,
    doctype_codes: list[str],
    max_results: int,
    page_size: int,
    view: str,
    sleep_sec: float,
) -> list[dict[str, Any]]:
    full_query = build_filtered_query(
        base_query=query_spec.query,
        min_year=min_year,
        language=language,
        doctype_codes=doctype_codes,
    )

    rows: list[dict[str, Any]] = []
    start = 0
    total = None

    while len(rows) < max_results:
        params = {
            "query": full_query,
            "start": start,
            "count": page_size,
            "view": view,
            "sort": "-coverDate",
        }
        payload = http_get_json(
            url=SCOPUS_SEARCH_URL,
            params=params,
            api_key=api_key,
            insttoken=insttoken,
        )
        sr = payload.get("search-results", {})
        entries = sr.get("entry") or []

        if total is None:
            total_raw = sr.get("opensearch:totalResults", "0")
            total = int(total_raw) if str(total_raw).isdigit() else 0

        if not entries:
            break

        for entry in entries:
            rows.append(parse_entry(query_spec.qid, full_query, entry))
            if len(rows) >= max_results:
                break

        start += len(entries)
        if total is not None and start >= total:
            break
        time.sleep(max(0.0, sleep_sec))

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = row.get("eid") or row.get("doi") or f"{row.get('title')}|{row.get('cover_date')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Scopus advanced metadata search")
    parser.add_argument("--output-dir", default="Review/output", help="Output directory")
    parser.add_argument("--min-year", type=int, default=2019, help="Minimum publication year")
    parser.add_argument("--language", default="english", help="Scopus language filter token")
    parser.add_argument(
        "--doctypes",
        default="ar,cp",
        help="Comma-separated Scopus doctype codes (e.g., ar,cp,cr)",
    )
    parser.add_argument("--max-results-per-query", type=int, default=300)
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--view", default="COMPLETE", choices=["STANDARD", "COMPLETE"])
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    args = parser.parse_args()

    api_key = os.getenv("ELS_API_KEY")
    insttoken = os.getenv("ELS_INSTTOKEN")
    if not api_key:
        raise RuntimeError("Environment variable ELS_API_KEY is required.")

    doctype_codes = [d.strip() for d in args.doctypes.split(",") if d.strip()]
    if not doctype_codes:
        raise RuntimeError("At least one doctype must be provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in DEFAULT_QUERIES:
        print(f"[{spec.qid}] {spec.description}")
        rows = search_scopus_query(
            query_spec=spec,
            api_key=api_key,
            insttoken=insttoken,
            min_year=args.min_year,
            language=args.language,
            doctype_codes=doctype_codes,
            max_results=args.max_results_per_query,
            page_size=args.page_size,
            view=args.view,
            sleep_sec=args.sleep_sec,
        )
        all_rows.extend(rows)

        q_csv = output_dir / f"{spec.qid}.csv"
        write_csv(q_csv, rows)
        summary_rows.append(
            {
                "query_id": spec.qid,
                "description": spec.description,
                "records": len(rows),
                "output_csv": str(q_csv),
            }
        )
        print(f"  -> {len(rows)} records")

    deduped = deduplicate_rows(all_rows)
    combined_csv = output_dir / "scopus_all_queries_combined.csv"
    dedup_csv = output_dir / "scopus_all_queries_dedup.csv"
    summary_csv = output_dir / "scopus_summary.csv"

    write_csv(combined_csv, all_rows)
    write_csv(dedup_csv, deduped)
    write_csv(summary_csv, summary_rows)

    print("\nDone.")
    print(f"Combined: {combined_csv} ({len(all_rows)})")
    print(f"Deduped : {dedup_csv} ({len(deduped)})")
    print(f"Summary : {summary_csv}")


if __name__ == "__main__":
    main()
