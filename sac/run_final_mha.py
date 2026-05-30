"""Final grid — MHA machine. Hardcoded config (no argparse).

Trains + tests annually all (method, tariff, seed) cells for MHA.
Skip-if-done automatic on both phases.

Run from project root:
    python sac/run_final_mha.py
"""
import sys
import time
from pathlib import Path

SAC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SAC_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sac.run_final_grid import build_cells, run_cell

# ---- machine config -----------------------------------------------------------
ARCH = "MHA"
METHODS = ["u_cmdp", "u_cmdp_ft", "u_cmdp_shaped", "u_penalty", "u_cmdp_droq", "u_cmdp_redq"]
TARIFFS = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]
BASE_SEEDS = [42]
EXTRA_SEEDS_ON_LEAD = [7, 13]
LEAD_TARIFF = "tar_sw"
SKIP_TEST = False
# ------------------------------------------------------------------------------


def main():
    cells = build_cells(METHODS, [ARCH], TARIFFS, BASE_SEEDS, EXTRA_SEEDS_ON_LEAD, LEAD_TARIFF)
    print(f"=== final grid | {ARCH} machine ===")
    print(f"methods: {METHODS}")
    print(f"tariffs: {TARIFFS}  (lead={LEAD_TARIFF} +seeds {EXTRA_SEEDS_ON_LEAD})")
    print(f"total cells: {len(cells)}  est ~{len(cells)*3.2:.0f}h sequential\n", flush=True)

    summary = []
    t_all = time.time()
    for i, (m, a, t, s) in enumerate(cells, 1):
        print(f"\n>>> [{i}/{len(cells)}] {m}/{a}/{t}-s{s}", flush=True)
        st = run_cell(m, a, t, s, SKIP_TEST)
        summary.append((m, a, t, s, st))
        print(f"<<< {m}/{a}/{t}-s{s}: {' | '.join(st)}  (total elapsed {(time.time()-t_all)/3600:.1f}h)", flush=True)

    print(f"\n=== {ARCH} grid complete: {(time.time()-t_all)/3600:.1f}h ===")
    for m, a, t, s, st in summary:
        print(f"  {m:<16} {a:<4} {t:<10} s{s:<3} : {' | '.join(st)}")


if __name__ == "__main__":
    main()
