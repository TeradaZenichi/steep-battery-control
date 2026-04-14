#!/usr/bin/env python3
"""
Web of Science metadata collector.

Supports:
- Expanded API (default): https://wos-api.clarivate.com/api/wos
- Starter API:           https://api.clarivate.com/apis/wos-starter/v1/documents

Usage example:
  set WOS_API_KEY=YOUR_KEY
  python Review/wos_advanced_search.py --api-mode expanded --max-results-per-query 300
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


WOS_EXPANDED_URL_DEFAULT = "https://wos-api.clarivate.com/api/wos"
WOS_STARTER_URL_DEFAULT = "https://api.clarivate.com/apis/wos-starter/v1/documents"


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


def ensure_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def get_text(x: Any, keys: list[str] | None = None) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, dict):
        if keys:
            for k in keys:
                v = x.get(k)
                txt = get_text(v)
                if txt:
                    return txt
        for k in ["content", "value", "full_name", "fullName", "display_name", "displayName", "name", "pref", "$"]:
            v = x.get(k)
            txt = get_text(v)
            if txt:
                return txt
    return ""


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
        "X-APIKey": api_key,
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


# -------------------------- Starter mode parsing --------------------------

def parse_starter_entry(query_id: str, query_text: str, hit: dict[str, Any]) -> dict[str, Any]:
    identifiers = hit.get("identifiers", {}) if isinstance(hit.get("identifiers"), dict) else {}
    names = hit.get("names", {}) if isinstance(hit.get("names"), dict) else {}
    source = hit.get("source", {}) if isinstance(hit.get("source"), dict) else {}
    links = hit.get("links", {}) if isinstance(hit.get("links"), dict) else {}
    citations = hit.get("citations", {}) if isinstance(hit.get("citations"), dict) else {}

    authors = []
    for author in ensure_list(names.get("authors")):
        txt = get_text(author, ["fullName", "displayName", "name"])
        if txt:
            authors.append(txt)

    affiliations = []
    for org in ensure_list(names.get("organizations")):
        txt = get_text(org, ["name", "pref", "content"])
        if txt:
            affiliations.append(txt)

    types = ensure_list(hit.get("types"))
    source_types = ensure_list(hit.get("sourceTypes"))

    return {
        "query_id": query_id,
        "query": query_text,
        "uid": get_text(hit, ["uid"]),
        "title": get_text(hit, ["title"]),
        "doi": get_text(identifiers, ["doi"]),
        "source_title": get_text(source, ["sourceTitle", "title"]),
        "publish_year": get_text(hit, ["publishYear", "year"]),
        "types": "; ".join([get_text(t) for t in types if get_text(t)]),
        "source_types": "; ".join([get_text(t) for t in source_types if get_text(t)]),
        "times_cited": get_text(citations, ["timesCited", "count"]),
        "authors": "; ".join(authors),
        "affiliations": "; ".join(affiliations),
        "record_link": get_text(links, ["record"]),
    }


def search_starter_query(
    endpoint: str,
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
        payload = http_get_json(url=endpoint, params=params, api_key=api_key)
        hits = ensure_list(payload.get("hits"))
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict) and total is None and isinstance(metadata.get("total"), int):
            total = int(metadata["total"])

        if not hits:
            break

        for hit in hits:
            if not isinstance(hit, dict):
                continue
            rows.append(parse_starter_entry(query_spec.qid, full_query, hit))
            if len(rows) >= max_results:
                break

        page += 1
        if total is not None and len(rows) >= total:
            break
        if len(hits) < page_size:
            break
        time.sleep(max(0.0, sleep_sec))

    return rows


# -------------------------- Expanded mode parsing -------------------------

def extract_expanded_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("Data", {})
    records_obj = {}
    if isinstance(data, dict):
        records_obj = data.get("Records", {}) or {}
    records = {}
    if isinstance(records_obj, dict):
        records = records_obj.get("records", {}) or {}
    rec = []
    if isinstance(records, dict):
        rec = ensure_list(records.get("REC"))
    elif isinstance(records, list):
        rec = records
    return [r for r in rec if isinstance(r, dict)]


def extract_expanded_total(payload: dict[str, Any]) -> int | None:
    q = payload.get("QueryResult", {})
    if not isinstance(q, dict):
        return None
    for key in ["RecordsFound", "recordsFound", "RecordsFoundInSubscription"]:
        val = q.get(key)
        txt = get_text(val)
        if txt.isdigit():
            return int(txt)
    return None


def parse_expanded_doi(rec: dict[str, Any]) -> str:
    ids = (
        rec.get("dynamic_data", {})
        .get("cluster_related", {})
        .get("identifiers", {})
        .get("identifier")
    )
    for ident in ensure_list(ids):
        if not isinstance(ident, dict):
            continue
        if get_text(ident.get("type")).lower() == "doi":
            doi = get_text(ident)
            if doi:
                return doi
    return ""


def parse_expanded_titles(summary: dict[str, Any]) -> tuple[str, str]:
    title_main = ""
    source_title = ""
    titles = summary.get("titles", {}).get("title")
    for t in ensure_list(titles):
        if not isinstance(t, dict):
            txt = get_text(t)
            if txt and not title_main:
                title_main = txt
            continue
        ttype = get_text(t.get("type")).lower()
        txt = get_text(t)
        if not txt:
            continue
        if ttype == "item" and not title_main:
            title_main = txt
        elif ttype in {"source", "source_abbrev"} and not source_title:
            source_title = txt
        elif not title_main:
            title_main = txt
    return title_main, source_title


def parse_expanded_authors(summary: dict[str, Any]) -> str:
    names = summary.get("names", {}).get("name")
    authors = []
    for n in ensure_list(names):
        if not isinstance(n, dict):
            continue
        role = get_text(n.get("role")).lower()
        if role and role != "author":
            continue
        nm = get_text(n, ["full_name", "fullName", "display_name", "displayName", "wos_standard"])
        if nm:
            authors.append(nm)
    return "; ".join(authors)


def parse_expanded_affiliations(rec: dict[str, Any]) -> str:
    aff = []
    addresses = (
        rec.get("static_data", {})
        .get("fullrecord_metadata", {})
        .get("addresses", {})
        .get("address_name")
    )
    for address in ensure_list(addresses):
        if not isinstance(address, dict):
            continue
        orgs = (
            address.get("address_spec", {})
            .get("organizations", {})
            .get("organization")
        )
        for org in ensure_list(orgs):
            txt = get_text(org, ["content", "pref", "name"])
            if txt:
                aff.append(txt)
    # deduplicate preserving order
    seen = set()
    out = []
    for a in aff:
        key = a.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return "; ".join(out)


def parse_expanded_types(summary: dict[str, Any]) -> str:
    doctypes = summary.get("doctypes", {}).get("doctype")
    vals = []
    for d in ensure_list(doctypes):
        txt = get_text(d)
        if txt:
            vals.append(txt)
    return "; ".join(vals)


def parse_expanded_source_types(summary: dict[str, Any]) -> str:
    pubtype = summary.get("pub_info", {}).get("pubtype")
    vals = []
    for p in ensure_list(pubtype):
        txt = get_text(p)
        if txt:
            vals.append(txt)
    return "; ".join(vals)


def parse_expanded_times_cited(rec: dict[str, Any]) -> str:
    silo = (
        rec.get("dynamic_data", {})
        .get("citation_related", {})
        .get("tc_list", {})
        .get("silo_tc")
    )
    candidates = [s for s in ensure_list(silo) if isinstance(s, dict)]
    if not candidates:
        return ""
    # prefer WOS/WOK aggregate
    ranked = []
    for c in candidates:
        coll = get_text(c.get("coll_id")).upper()
        count = get_text(c.get("local_count"))
        ranked.append((0 if coll in {"WOS", "WOK"} else 1, count))
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1] if ranked else ""


def parse_expanded_entry(query_id: str, query_text: str, rec: dict[str, Any]) -> dict[str, Any]:
    summary = rec.get("static_data", {}).get("summary", {})
    title, source_title = parse_expanded_titles(summary)

    uid = get_text(rec.get("UID"))
    record_link = f"https://www.webofscience.com/wos/woscc/full-record/{uid}" if uid else ""

    return {
        "query_id": query_id,
        "query": query_text,
        "uid": uid,
        "title": title,
        "doi": parse_expanded_doi(rec),
        "source_title": source_title,
        "publish_year": get_text(summary.get("pub_info", {}).get("pubyear")),
        "types": parse_expanded_types(summary),
        "source_types": parse_expanded_source_types(summary),
        "times_cited": parse_expanded_times_cited(rec),
        "authors": parse_expanded_authors(summary),
        "affiliations": parse_expanded_affiliations(rec),
        "record_link": record_link,
    }


def search_expanded_query(
    endpoint: str,
    query_spec: QuerySpec,
    api_key: str,
    db: str,
    min_year: int,
    max_year: int,
    language: str,
    doc_types: list[str],
    max_results: int,
    page_size: int,
    option_view: str,
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
    first_record = 1
    total = None

    while len(rows) < max_results:
        params = {
            "databaseId": db,
            "usrQuery": full_query,
            "count": page_size,
            "firstRecord": first_record,
            "optionView": option_view,
        }
        payload = http_get_json(url=endpoint, params=params, api_key=api_key)
        recs = extract_expanded_records(payload)
        if total is None:
            total = extract_expanded_total(payload)

        if not recs:
            break

        for rec in recs:
            rows.append(parse_expanded_entry(query_spec.qid, full_query, rec))
            if len(rows) >= max_results:
                break

        first_record += len(recs)
        if total is not None and len(rows) >= total:
            break
        if len(recs) < page_size:
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
    parser = argparse.ArgumentParser(description="Web of Science advanced metadata search")
    parser.add_argument("--output-dir", default="Review/output_wos", help="Output directory")
    parser.add_argument("--api-mode", default="expanded", choices=["expanded", "starter"])
    parser.add_argument("--expanded-endpoint", default=WOS_EXPANDED_URL_DEFAULT)
    parser.add_argument("--starter-endpoint", default=WOS_STARTER_URL_DEFAULT)
    parser.add_argument("--db", default="WOS", help="WOS database code")
    parser.add_argument("--min-year", type=int, default=2019, help="Minimum publication year")
    parser.add_argument("--max-year", type=int, default=2026, help="Maximum publication year")
    parser.add_argument("--language", default="English", help="WoS language label")
    parser.add_argument("--doc-types", default="Article", help="Comma-separated WoS document types")
    parser.add_argument("--max-results-per-query", type=int, default=300)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--detail", default="full", choices=["full", "short"], help="Starter API only")
    parser.add_argument("--option-view", default="SR", choices=["SR", "FR"], help="Expanded API only")
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    args = parser.parse_args()

    if args.api_mode == "starter":
        if args.page_size < 1 or args.page_size > 50:
            raise RuntimeError("For WoS Starter, --page-size must be between 1 and 50.")
    else:
        if args.page_size < 1 or args.page_size > 100:
            raise RuntimeError("For WoS Expanded, --page-size must be between 1 and 100.")

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
        if args.api_mode == "starter":
            rows = search_starter_query(
                endpoint=args.starter_endpoint,
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
        else:
            rows = search_expanded_query(
                endpoint=args.expanded_endpoint,
                query_spec=spec,
                api_key=api_key,
                db=args.db,
                min_year=args.min_year,
                max_year=args.max_year,
                language=args.language,
                doc_types=doc_types,
                max_results=args.max_results_per_query,
                page_size=args.page_size,
                option_view=args.option_view,
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
