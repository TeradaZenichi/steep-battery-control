#!/usr/bin/env python3
"""
Web of Science Starter API advanced-search metadata collector.

Usage example:
  set WOS_API_KEY=YOUR_KEY
  python Review/wos_advanced_search.py --max-results-per-query 300 --detail full
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


WOS_STARTER_URL = "https://api.clarivate.com/apis/wos-starter/v1/documents"


@dataclass(frozen=True)
class QuerySpec:
    qid: str
    description: str
    query_topic: str


DEFAULT_QUERIES: list[QuerySpec] = [
    QuerySpec(
        qid="Q1_hems_der_tariffs",
        description="HEMS with PV+BESS+EV under dynamic tariffs",
        query_topic='TS=("home energy management" OR HEMS OR "residential energy management") AND TS=(PV OR photovoltaic) AND TS=(battery OR BESS) AND TS=(EV OR "electric vehicle") AND TS=("dynamic tariff" OR "time-of-use" OR "real-time pricing")',
    ),
    QuerySpec(
        qid="Q2_milp_mpc_online",
        description="Classical optimization baselines",
        query_topic='TS=(HEMS OR "residential energy management") AND TS=(MILP OR "mixed-integer linear programming" OR MPC OR "model predictive control") AND TS=("real-time" OR online)',
    ),
    QuerySpec(
        qid="Q3_rl_energy",
        description="RL for energy management",
        query_topic='TS=("reinforcement learning" OR "deep reinforcement learning") AND TS=("home energy management" OR microgrid) AND TS=(PV OR battery OR EV)',
    ),
    QuerySpec(
        qid="Q4_offpolicy_vs_onpolicy",
        description="SAC and other RL families",
        query_topic='TS=("soft actor-critic" OR SAC OR TD3 OR DDPG OR PPO) AND TS=("energy management" OR microgrid OR HEMS)',
    ),
    QuerySpec(
        qid="Q5_safe_rl_constraints",
        description="Safe/Constrained RL and action projection",
        query_topic='TS=("safe reinforcement learning" OR "constrained reinforcement learning" OR "action projection" OR "safety layer") AND TS=("energy management" OR microgrid OR "power systems")',
    ),
    QuerySpec(
        qid="Q6_pomdp_temporal_models",
        description="Partial observability and temporal architectures",
        query_topic='TS=(POMDP OR "partial observability") AND TS=("reinforcement learning") AND TS=(GRU OR LSTM OR TCN OR Transformer OR Attention)',
    ),
    QuerySpec(
        qid="Q7_il_plus_rl",
        description="Imitation learning + RL hybrid pipelines",
        query_topic='TS=("imitation learning" OR "behavior cloning") AND TS=("reinforcement learning") AND TS=("warm start" OR pretraining OR initialization) AND TS=(energy OR microgrid OR HEMS)',
    ),
    QuerySpec(
        qid="Q8_multi_tariff_generalization",
        description="Tariff-aware robustness/generalization",
        query_topic='TS=("dynamic pricing" OR "time-of-use" OR tariff) AND TS=("energy management") AND TS=(robustness OR generalization OR "multi-tariff")',
    ),
]


def getenv_required(name: str) -> str:
    import os

    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required.")
    return value


def build_filtered_query(
    base_topic_query: str,
    min_year: int,
    max_year: int,
    language: str,
    doc_types: list[str],
) -> str:
    doctype_clause = " OR ".join(doc_types)
    return (
        f"({base_topic_query}) AND PY=({min_year}-{max_year}) "
        f"AND LA=({language}) AND DT=({doctype_clause})"
    )


def http_get_json(
    url: str,
    params: dict[str, Any],
    api_key: str,
    retries: int = 5,
    timeout_sec: int = 60,
) -> dict[str, Any]:
    query_string = urlencode(params)
    full_url = f"{url}?{query_string}"
    headers = {
        "Accept": "application/json",
        "X-ApiKey": api_key,
    }

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


def _first(data: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def parse_entry(query_id: str, query_text: str, hit: dict[str, Any]) -> dict[str, Any]:
    identifiers = hit.get("identifiers", {}) if isinstance(hit.get("identifiers"), dict) else {}
    names = hit.get("names", {}) if isinstance(hit.get("names"), dict) else {}
    source = hit.get("source", {}) if isinstance(hit.get("source"), dict) else {}
    links = hit.get("links", {}) if isinstance(hit.get("links"), dict) else {}
    citations = hit.get("citations", {}) if isinstance(hit.get("citations"), dict) else {}

    author_list = []
    authors = names.get("authors", [])
    if isinstance(authors, list):
        for author in authors:
            if isinstance(author, dict):
                full_name = _first(author, ["fullName", "displayName", "name"])
                if full_name:
                    author_list.append(full_name)
            elif isinstance(author, str) and author.strip():
                author_list.append(author.strip())

    affiliation_list = []
    affiliations = names.get("organizations", [])
    if isinstance(affiliations, list):
        for org in affiliations:
            if isinstance(org, dict):
                org_name = _first(org, ["name", "pref", "content"])
                if org_name:
                    affiliation_list.append(org_name)
            elif isinstance(org, str) and org.strip():
                affiliation_list.append(org.strip())

    types = hit.get("types", [])
    source_types = hit.get("sourceTypes", [])
    if not isinstance(types, list):
        types = [str(types)]
    if not isinstance(source_types, list):
        source_types = [str(source_types)]

    return {
        "query_id": query_id,
        "query": query_text,
        "uid": _first(hit, ["uid"]),
        "title": _first(hit, ["title"]),
        "doi": _first(identifiers, ["doi"]),
        "source_title": _first(source, ["sourceTitle", "title"]),
        "publish_year": _first(hit, ["publishYear", "year"]),
        "types": "; ".join([t for t in types if t]),
        "source_types": "; ".join([t for t in source_types if t]),
        "times_cited": _first(citations, ["timesCited", "count"]),
        "authors": "; ".join(author_list),
        "affiliations": "; ".join(affiliation_list),
        "record_link": _first(links, ["record"]),
    }


def search_wos_query(
    query_spec: QuerySpec,
    api_key: str,
    db: str,
    min_year: int,
    max_year: int,
    language: str,
    doc_types: list[str],
    max_results: int,
    page_size: int,
    detail: str,
    sleep_sec: float,
) -> list[dict[str, Any]]:
    full_query = build_filtered_query(
        base_topic_query=query_spec.query_topic,
        min_year=min_year,
        max_year=max_year,
        language=language,
        doc_types=doc_types,
    )

    rows: list[dict[str, Any]] = []
    page = 1
    total = None

    while len(rows) < max_results:
        params = {
            "db": db,
            "q": full_query,
            "limit": page_size,
            "page": page,
            "detail": detail,
        }
        payload = http_get_json(
            url=WOS_STARTER_URL,
            params=params,
            api_key=api_key,
        )

        hits = payload.get("hits", [])
        if not isinstance(hits, list):
            hits = []

        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict):
            total_raw = metadata.get("total")
            if total is None and isinstance(total_raw, int):
                total = total_raw

        if not hits:
            break

        for hit in hits:
            if not isinstance(hit, dict):
                continue
            rows.append(parse_entry(query_spec.qid, full_query, hit))
            if len(rows) >= max_results:
                break

        page += 1
        if total is not None and len(rows) >= total:
            break
        if len(hits) < page_size:
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
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = row.get("uid") or row.get("doi") or f"{row.get('title')}|{row.get('publish_year')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Web of Science Starter advanced metadata search")
    parser.add_argument("--output-dir", default="Review/output_wos", help="Output directory")
    parser.add_argument("--db", default="WOS", help="WOS database code")
    parser.add_argument("--min-year", type=int, default=2019, help="Minimum publication year")
    parser.add_argument("--max-year", type=int, default=2026, help="Maximum publication year")
    parser.add_argument("--language", default="English", help="WoS language label")
    parser.add_argument(
        "--doc-types",
        default="Article,Proceedings Paper",
        help="Comma-separated WoS document types",
    )
    parser.add_argument("--max-results-per-query", type=int, default=300)
    parser.add_argument("--page-size", type=int, default=50, help="WoS Starter supports up to 50")
    parser.add_argument("--detail", default="full", choices=["full", "short"])
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    args = parser.parse_args()

    if args.page_size < 1 or args.page_size > 50:
        raise RuntimeError("For WoS Starter, --page-size must be between 1 and 50.")

    api_key = getenv_required("WOS_API_KEY")
    doc_types = [d.strip() for d in args.doc_types.split(",") if d.strip()]
    if not doc_types:
        raise RuntimeError("At least one doc type must be provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in DEFAULT_QUERIES:
        print(f"[{spec.qid}] {spec.description}")
        rows = search_wos_query(
            query_spec=spec,
            api_key=api_key,
            db=args.db,
            min_year=args.min_year,
            max_year=args.max_year,
            language=args.language,
            doc_types=doc_types,
            max_results=args.max_results_per_query,
            page_size=args.page_size,
            detail=args.detail,
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
    combined_csv = output_dir / "wos_all_queries_combined.csv"
    dedup_csv = output_dir / "wos_all_queries_dedup.csv"
    summary_csv = output_dir / "wos_summary.csv"

    write_csv(combined_csv, all_rows)
    write_csv(dedup_csv, deduped)
    write_csv(summary_csv, summary_rows)

    print("\nDone.")
    print(f"Combined: {combined_csv} ({len(all_rows)})")
    print(f"Deduped : {dedup_csv} ({len(deduped)})")
    print(f"Summary : {summary_csv}")


if __name__ == "__main__":
    main()
