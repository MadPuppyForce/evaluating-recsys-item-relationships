#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: (generated helper)
# Date: 2026
#
# Erratum companion: compare Usefulness and ILR AS PUBLISHED (single-orientation
# relation lookup, release v1.0-paper) with the CORRECTED (symmetric) lookup.
#
# WHY
# ---
# Up to release v1.0-paper, `max_relation_score(a, b)` in metrics/usefulness_ilr.py
# looked up a single key orientation:
#       max(subst_map.get((a, b), 0.0), compl_map.get((a, b), 0.0))
# while `build_relation_maps` stores each pair only once, as (str(min_id), str(max_id))
# (because the score CSVs have item_x < item_y). Consequently:
#   * ILR: combinations() follow the bundle's list order (NOT sorted by id), so a
#     pair listed "larger-id first" missed the map and read 0.
#   * Usefulness: u_qr queries (input, output) and u_rq queries (output, input),
#     i.e. opposite orientations against a one-way map, so each real pair was seen
#     by only ONE of the two directional terms.
# The paper defines substitutability and complementarity as SYMMETRIC relations,
# so the intended score for a pair {a, b} is orientation-independent.
# metrics/usefulness_ilr.py now queries both orientations.
#
# WHAT THIS SCRIPT DOES
# ---------------------
# It re-runs the EXACT experiment protocol (same splits / seed / R / corruption
# draws), computing for every corrupted sample BOTH:
#   - *_ref : the published behaviour (single-orientation lookup, kept in this
#             file as a frozen copy of the v1.0-paper code)
#   - *_sym : the corrected behaviour (usefulness()/ilr() imported from
#             metrics/usefulness_ilr.py)
# then reports Spearman(rho, metric) and the total drop Delta for each, per domain
# and globally.
#
# HOW TO RUN
# ----------
#   Requires the score CSVs produced by run_scores_experiment.py, i.e.
#       results/<domain>/substitutability_scores.csv
#       results/<domain>/complementarity_scores.csv
#   plus data/raw/<domain>/session_item.csv and data/processed/<domain>/bundle_item_test.csv
#   Then simply:  python scripts/recompute_symmetric_usefulness_ilr.py
#   (paths are resolved relative to this file, so the working directory does not matter)

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple
from itertools import combinations
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# --- make `metrics.usefulness_ilr` importable regardless of cwd --------------
HERE = Path(__file__).resolve().parent          # .../scripts
ROOT = HERE.parent                               # .../AIDT 2026
sys.path.insert(0, str(ROOT / "src"))

# Import the CORRECTED implementation (used as the "sym" side).
from metrics.usefulness_ilr import (            # noqa: E402
    build_relation_maps,
    usefulness as usefulness_sym,
    ilr as ilr_sym,
)


# =========================
# CONFIG  (identical to run_metrics_experiment.py)
# =========================
RAW_PATH = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed"
RESULTS_PATH = ROOT / "results"

DOMAINS = ["electronic", "clothing", "food"]

R = 100
SEED = 42
K_MIN = 2
K_MAX = 10
INPUT_RATIO = 0.6


# =========================
# Published (v1.0-paper) relation lookup -- frozen copy, do NOT fix
# =========================
def ref_relation(a: str, b: str, subst_map: Dict, compl_map: Dict) -> float:
    """max(subst, compl) queried in ONE key orientation only, as published.

    The maps store each pair once, so a pair presented in the other orientation
    reads 0.0. This is the behaviour that produced the figures of the paper.
    """
    return max(subst_map.get((a, b), 0.0), compl_map.get((a, b), 0.0))


def _dir_usefulness_ref(source, target, subst_map, compl_map) -> float:
    if not source or not target:
        return 0.0
    total = 0.0
    for src in source:
        best = 0.0
        for tgt in target:
            best = max(best, ref_relation(src, tgt, subst_map, compl_map))
        total += best
    return total / float(len(source))


def usefulness_ref(session_items, recommended_items, subst_map, compl_map) -> float:
    if not session_items or not recommended_items:
        return 0.0
    u_qr = _dir_usefulness_ref(session_items, recommended_items, subst_map, compl_map)
    u_rq = _dir_usefulness_ref(recommended_items, session_items, subst_map, compl_map)
    denom = u_qr + u_rq
    return (2.0 * u_qr * u_rq) / denom if denom != 0.0 else 0.0


def ilr_ref(recommended_items, subst_map, compl_map) -> float:
    k = len(recommended_items)
    if k < 2:
        return 0.0
    total = 0.0
    for item_a, item_b in combinations(recommended_items, 2):
        total += ref_relation(item_a, item_b, subst_map, compl_map)
    return (2.0 / (k * (k - 1))) * total


# =========================
# Data loading  (copied from run_metrics_experiment.py)
# =========================
def _read_csv(path: Path, sep: str = ",") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, sep=sep)


def _ensure_cols(df: pd.DataFrame, required: List[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns {missing}. Found={list(df.columns)}")


def _domain_title(domain: str) -> str:
    return {"electronic": "Electronic", "clothing": "Clothing", "food": "Food"}.get(domain, domain)


def _load_scores(domain: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subst_df = _read_csv(RESULTS_PATH / domain / "substitutability_scores.csv")
    compl_df = _read_csv(RESULTS_PATH / domain / "complementarity_scores.csv")
    _ensure_cols(subst_df, ["item_x", "item_y"], "subst_df")
    _ensure_cols(compl_df, ["item_x", "item_y"], "compl_df")
    return subst_df, compl_df


def _load_catalogue_items(domain: str) -> np.ndarray:
    df = _read_csv(RAW_PATH / domain / "session_item.csv")
    _ensure_cols(df, ["item ID"], "session_item (raw)")
    return df["item ID"].unique()


def _load_test_bundles(domain: str) -> pd.DataFrame:
    df = _read_csv(PROCESSED_PATH / domain / "bundle_item_test.csv")
    _ensure_cols(df, ["bundle ID", "item ID"], "bundle_item_test")
    tb = (
        df.groupby("bundle ID")["item ID"].apply(list).reset_index().rename(columns={"item ID": "items"})
    )
    tb["bundle_size"] = tb["items"].apply(len)
    return tb


# =========================
# Corruption  (copied verbatim from run_metrics_experiment.py)
# =========================
def corrupt_bundle(items, m, catalogue, rng):
    items_list = list(items)
    k = len(items_list)
    if m < 0 or m > k:
        raise ValueError("m must be in [0, k].")
    if m == 0:
        return items_list.copy()
    positions = rng.choice(k, size=m, replace=False)
    available = np.array([x for x in catalogue if x not in items_list], dtype=object)
    if len(available) < m:
        raise ValueError("Catalogue too small to draw m distinct intruders.")
    replacements = rng.choice(available, size=m, replace=False)
    corrupted = items_list.copy()
    for pos, rep in zip(positions, replacements):
        corrupted[pos] = rep
    return corrupted


def split_bundle(items, ratio=0.6):
    items = list(items)
    k = len(items)
    input_size = max(1, int(math.floor(ratio * k)))
    output_size = k - input_size
    if input_size < 1 or output_size < 1:
        raise ValueError(f"Invalid split: input={input_size}, output={output_size}, k={k}")
    return items[:input_size], items[input_size:]


def corrupt_output(inputs, outputs, m, catalogue, rng):
    inputs = list(inputs)
    outputs = list(outputs)
    o = len(outputs)
    if m < 0 or m > o:
        raise ValueError("m must be in [0, |O|].")
    if m == 0:
        return outputs.copy()
    positions = rng.choice(o, size=m, replace=False)
    forbidden = set(inputs) | set(outputs)
    candidates = np.array([x for x in catalogue if x not in forbidden], dtype=object)
    if len(candidates) < m:
        raise ValueError("Catalogue too small to draw m distinct intruders.")
    replacements = rng.choice(candidates, size=m, replace=False)
    corrupted = outputs.copy()
    for pos, rep in zip(positions, replacements):
        corrupted[pos] = rep
    return corrupted


# =========================
# Degradation indicators
# =========================
def _spearman_and_drop(curve: pd.DataFrame, rho_col: str, val_col: str) -> Tuple[float, float]:
    sp = spearmanr(curve[rho_col], curve[val_col]).correlation if len(curve) >= 2 else float("nan")
    mu0 = curve.loc[np.isclose(curve[rho_col], 0.0), val_col].mean()
    mu1 = curve.loc[np.isclose(curve[rho_col], 1.0), val_col].mean()
    drop = mu0 - mu1 if not (np.isnan(mu0) or np.isnan(mu1)) else float("nan")
    return sp, drop


def _bundle_order_diagnostic(test_bundles: pd.DataFrame) -> Tuple[float, float]:
    """Fraction of test bundles not sorted by id, and mean fraction of adjacent
    descending pairs (a proxy for how many ILR pairs are queried wrong-way)."""
    unsorted, desc_frac = [], []
    for items in test_bundles["items"]:
        lst = list(items)
        unsorted.append(lst != sorted(lst))
        pairs = list(zip(lst, lst[1:]))
        if pairs:
            desc_frac.append(sum(a > b for a, b in pairs) / len(pairs))
    return (float(np.mean(unsorted)) if unsorted else float("nan"),
            float(np.mean(desc_frac)) if desc_frac else float("nan"))


# =========================
# Per-domain run
# =========================
def run_domain(domain: str) -> Dict[str, Any]:
    print(f"\n==================== DOMAIN: {domain} ====================")
    subst_df, compl_df = _load_scores(domain)
    subst_map, compl_map = build_relation_maps(subst_df, compl_df)
    catalogue = _load_catalogue_items(domain)
    test_bundles = _load_test_bundles(domain)

    frac_unsorted, frac_desc = _bundle_order_diagnostic(test_bundles)
    print(f"[diag] test bundles not sorted by id: {frac_unsorted*100:.1f}%  |  "
          f"mean adjacent-descending pairs: {frac_desc*100:.1f}%")

    available_k = sorted(test_bundles["bundle_size"].unique())
    k_values = [k for k in range(K_MIN, K_MAX + 1) if k in available_k]
    if not k_values:
        raise RuntimeError(f"No bundle sizes in [{K_MIN},{K_MAX}] for {domain}. Available={available_k}")

    rng = np.random.default_rng(SEED)   # SAME rng lifecycle as the original script

    # -------- ILR (same iteration order as generate_ilr_runs_for_k_range) ----
    ilr_records: List[Dict[str, Any]] = []
    for k in k_values:
        subset = test_bundles[test_bundles["bundle_size"] == k]
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            items = row["items"]
            bundle_id = row["bundle ID"]
            for m in range(k + 1):
                for _r in range(R):
                    corrupted = corrupt_bundle(items, m, catalogue, rng)
                    recs = [str(x) for x in corrupted]
                    ilr_records.append({
                        "bundle_id": bundle_id, "bundle_size": k, "m_corruption": m,
                        "ilr_ref": ilr_ref(recs, subst_map=subst_map, compl_map=compl_map),
                        "ilr_sym": ilr_sym(recs, subst_map, compl_map),
                    })
    df_ilr = pd.DataFrame(ilr_records)
    agg_ilr = (df_ilr.groupby(["bundle_id", "bundle_size", "m_corruption"])
               .agg(mean_ref=("ilr_ref", "mean"), mean_sym=("ilr_sym", "mean")).reset_index())
    agg_ilr["rho"] = agg_ilr["m_corruption"] / agg_ilr["bundle_size"]
    ilr_curve = agg_ilr.groupby("rho").agg(mean_ref=("mean_ref", "mean"),
                                           mean_sym=("mean_sym", "mean")).reset_index().sort_values("rho")
    (RESULTS_PATH / domain).mkdir(parents=True, exist_ok=True)
    agg_ilr.to_csv(RESULTS_PATH / domain / "ilr_agg_symcompare.csv", index=False)

    # -------- Usefulness (same iteration order as the original loop) ---------
    use_records: List[Dict[str, Any]] = []
    for _, row in test_bundles.iterrows():
        items = row["items"]
        inputs, outputs = split_bundle(items, ratio=INPUT_RATIO)
        o = len(outputs)
        inp = [str(x) for x in inputs]
        for m in range(o + 1):
            for _r in range(R):
                corrupted_o = corrupt_output(inputs, outputs, m, catalogue, rng)
                out = [str(x) for x in corrupted_o]
                use_records.append({
                    "bundle_id": row["bundle ID"], "m_corruption": m, "output_size": o,
                    "use_ref": usefulness_ref(inp, out, subst_map=subst_map, compl_map=compl_map),
                    "use_sym": usefulness_sym(inp, out, subst_map, compl_map),
                })
    df_use = pd.DataFrame(use_records)
    agg_use = (df_use.groupby(["bundle_id", "m_corruption", "output_size"])
               .agg(mean_ref=("use_ref", "mean"), mean_sym=("use_sym", "mean")).reset_index())
    agg_use["rho_out"] = agg_use["m_corruption"] / agg_use["output_size"]
    use_curve = agg_use.groupby("rho_out").agg(mean_ref=("mean_ref", "mean"),
                                               mean_sym=("mean_sym", "mean")).reset_index().sort_values("rho_out")
    agg_use.to_csv(RESULTS_PATH / domain / "usefulness_agg_symcompare.csv", index=False)

    # -------- indicators -----------------------------------------------------
    sp_ilr_ref, d_ilr_ref = _spearman_and_drop(ilr_curve, "rho", "mean_ref")
    sp_ilr_sym, d_ilr_sym = _spearman_and_drop(ilr_curve, "rho", "mean_sym")
    sp_use_ref, d_use_ref = _spearman_and_drop(use_curve, "rho_out", "mean_ref")
    sp_use_sym, d_use_sym = _spearman_and_drop(use_curve, "rho_out", "mean_sym")

    def _line(name, sp_r, sp_s, d_r, d_s, mean_r, mean_s):
        gap = (mean_s - mean_r) / mean_r * 100 if mean_r else float("nan")
        print(f"  {name:12s} | Spearman ref={sp_r:+.4f} sym={sp_s:+.4f} "
              f"| Drop ref={d_r:.5f} sym={d_s:.5f} "
              f"| mean ref={mean_r:.5f} sym={mean_s:.5f} (+{gap:.0f}%)")

    print("[results]")
    _line("ILR", sp_ilr_ref, sp_ilr_sym, d_ilr_ref, d_ilr_sym,
          ilr_curve["mean_ref"].mean(), ilr_curve["mean_sym"].mean())
    _line("Usefulness", sp_use_ref, sp_use_sym, d_use_ref, d_use_sym,
          use_curve["mean_ref"].mean(), use_curve["mean_sym"].mean())

    return {
        "domain": domain,
        "ilr_agg": agg_ilr.assign(subset=domain),
        "use_agg": agg_use.assign(subset=domain),
        "summary": [
            {"domain": _domain_title(domain), "metric": "ILR",
             "spearman_ref": sp_ilr_ref, "spearman_sym": sp_ilr_sym,
             "drop_ref": d_ilr_ref, "drop_sym": d_ilr_sym,
             "mean_ref": ilr_curve["mean_ref"].mean(), "mean_sym": ilr_curve["mean_sym"].mean()},
            {"domain": _domain_title(domain), "metric": "Usefulness",
             "spearman_ref": sp_use_ref, "spearman_sym": sp_use_sym,
             "drop_ref": d_use_ref, "drop_sym": d_use_sym,
             "mean_ref": use_curve["mean_ref"].mean(), "mean_sym": use_curve["mean_sym"].mean()},
        ],
    }


def aggregate_global(domain_outputs: List[Dict[str, Any]]) -> None:
    print(f"\n==================== GLOBAL ====================")
    ilr_all = pd.concat([d["ilr_agg"] for d in domain_outputs], ignore_index=True)
    use_all = pd.concat([d["use_agg"] for d in domain_outputs], ignore_index=True)
    ilr_all["bid"] = ilr_all["subset"] + "_" + ilr_all["bundle_id"].astype(str)
    use_all["bid"] = use_all["subset"] + "_" + use_all["bundle_id"].astype(str)

    ilr_curve = ilr_all.groupby("rho").agg(mean_ref=("mean_ref", "mean"),
                                           mean_sym=("mean_sym", "mean")).reset_index().sort_values("rho")
    use_curve = use_all.groupby("rho_out").agg(mean_ref=("mean_ref", "mean"),
                                               mean_sym=("mean_sym", "mean")).reset_index().sort_values("rho_out")

    sp_ilr_ref, d_ilr_ref = _spearman_and_drop(ilr_curve, "rho", "mean_ref")
    sp_ilr_sym, d_ilr_sym = _spearman_and_drop(ilr_curve, "rho", "mean_sym")
    sp_use_ref, d_use_ref = _spearman_and_drop(use_curve, "rho_out", "mean_ref")
    sp_use_sym, d_use_sym = _spearman_and_drop(use_curve, "rho_out", "mean_sym")

    print(f"  ILR        | Spearman ref={sp_ilr_ref:+.4f} sym={sp_ilr_sym:+.4f} "
          f"| Drop ref={d_ilr_ref:.5f} sym={d_ilr_sym:.5f}")
    print(f"  Usefulness | Spearman ref={sp_use_ref:+.4f} sym={sp_use_sym:+.4f} "
          f"| Drop ref={d_use_ref:.5f} sym={d_use_sym:.5f}")

    rows = [r for d in domain_outputs for r in d["summary"]]
    rows.append({"domain": "GLOBAL", "metric": "ILR",
                 "spearman_ref": sp_ilr_ref, "spearman_sym": sp_ilr_sym,
                 "drop_ref": d_ilr_ref, "drop_sym": d_ilr_sym,
                 "mean_ref": ilr_curve["mean_ref"].mean(), "mean_sym": ilr_curve["mean_sym"].mean()})
    rows.append({"domain": "GLOBAL", "metric": "Usefulness",
                 "spearman_ref": sp_use_ref, "spearman_sym": sp_use_sym,
                 "drop_ref": d_use_ref, "drop_sym": d_use_sym,
                 "mean_ref": use_curve["mean_ref"].mean(), "mean_sym": use_curve["mean_sym"].mean()})
    out = RESULTS_PATH / "symcompare_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved summary: {out}")
    print("Per-domain per-bundle tables: results/<domain>/{ilr,usefulness}_agg_symcompare.csv")


def main() -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    outputs = [run_domain(d) for d in DOMAINS]
    aggregate_global(outputs)
    print("\nDone. 'ref' = published (v1.0-paper) lookup, 'sym' = corrected symmetric lookup.")


if __name__ == "__main__":
    main()