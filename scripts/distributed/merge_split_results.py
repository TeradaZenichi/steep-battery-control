from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def copy_tree(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0

    copied = 0
    for child in src.rglob("*"):
        if child.is_file():
            rel = child.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge transferred split results into local Results/")
    parser.add_argument("--source", required=True, help="Path to .zip package or extracted folder containing Results/")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    copied = 0

    if source.is_file() and source.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(tmp)
            src_results = tmp / "Results"
            dst_results = ROOT / "Results"
            copied += copy_tree(src_results, dst_results)
    else:
        src_results = source / "Results" if (source / "Results").exists() else source
        dst_results = ROOT / "Results"
        copied += copy_tree(src_results, dst_results)

    print(f"[MERGE OK] copied_files={copied}")


if __name__ == "__main__":
    main()
