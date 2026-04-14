from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ASSIGNMENTS = {
    "A": ["TCN", "ATT_MEM", "ATT_MEMv2", "ATTv2", "MLPv2"],
    "B": ["TCNv2", "ATT", "GRU", "MLP", "GRUv2"],
}


def add_path(zf: zipfile.ZipFile, path: Path) -> int:
    count = 0
    if not path.exists():
        return count

    if path.is_file():
        zf.write(path, path.relative_to(ROOT))
        return 1

    for child in path.rglob("*"):
        if child.is_file():
            zf.write(child, child.relative_to(ROOT))
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Package split training results for transfer")
    parser.add_argument("--machine", choices=["A", "B"], required=True)
    parser.add_argument("--stage", choices=["all", "il", "rl"], default="all")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out = Path(args.output)
    else:
        out = ROOT / "Results" / "analysis" / f"package_machine_{args.machine}_{args.stage}_{ts}.zip"

    out.parent.mkdir(parents=True, exist_ok=True)

    models = ASSIGNMENTS[args.machine]
    include_il = args.stage in {"all", "il"}
    include_rl = args.stage in {"all", "rl"}

    manifest = {
        "machine": args.machine,
        "stage": args.stage,
        "models": models,
        "included": [],
    }

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        file_count = 0

        for model in models:
            if include_il:
                p = ROOT / "Results" / "train" / model / "1-IL"
                n = add_path(zf, p)
                if n:
                    manifest["included"].append(str(p.relative_to(ROOT)))
                    file_count += n

            if include_rl:
                p = ROOT / "Results" / "train" / model / "2-RL"
                n = add_path(zf, p)
                if n:
                    manifest["included"].append(str(p.relative_to(ROOT)))
                    file_count += n

        distributed_analysis = ROOT / "Results" / "analysis" / "distributed"
        n = add_path(zf, distributed_analysis)
        if n:
            manifest["included"].append(str(distributed_analysis.relative_to(ROOT)))
            file_count += n

        manifest_path = ROOT / "Results" / "analysis" / f"package_manifest_machine_{args.machine}_{args.stage}_{ts}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        zf.write(manifest_path, manifest_path.relative_to(ROOT))
        file_count += 1

    print(f"[SAVED] {out}")
    print(f"[FILES] {file_count}")


if __name__ == "__main__":
    main()
