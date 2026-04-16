'''
Goal: Apply the PC algorithm on the simulated Asia dataset
Author: Saptarshi Pyne
Date: April 14, 2026

'''

# -------------------------------------------------------
## Begin: Import modules
# -------------------------------------------------------
import os
import time
import pickle
import argparse
import tracemalloc
import datetime
from itertools import product

import psutil

## loadData.py
from loadData import loadData

from pgmpy.causal_discovery import PC
# -------------------------------------------------------
## End: Import modules
# -------------------------------------------------------


# -------------------------------------------------------
## Helper: evaluate structure
# -------------------------------------------------------
def evaluate_structure(est_dag, true_model, data, scoring_method="bic"):
    """Compute F1, SHD between estimated and true DAG."""
    true_edges = set(true_model.edges())
    est_edges  = set(est_dag.edges())

    tp = len(true_edges & est_edges)
    fp = len(est_edges - true_edges)
    fn = len(true_edges - est_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Structural Hamming Distance
    shd = fp + fn

    return {"f1": f1, "precision": precision, "recall": recall, "shd": shd}


# -------------------------------------------------------
## Parse CLI arguments
# -------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PC algorithm on the Asia dataset."
    )
    parser.add_argument(
        "--significance_level",
        type=float,
        nargs="+",
        default=[0.01, 0.05],
        help="One or more alpha values to sweep. Default: 0.01 0.05",
    )
    parser.add_argument(
        "--max_cond_vars",
        type=int,
        default=5,
        help="Maximum number of conditioning variables in CI tests. Default: 5",
    )
    parser.add_argument(
        "--expert_knowledge",
        type=str,
        default=None,
        help=(
            "Path to a pickle file containing expert knowledge "
            "(pgmpy DAGModel). Default: None"
        ),
    )
    parser.add_argument(
        "--enforce_expert_knowledge",
        action="store_true",
        default=False,
        help="Enforce expert knowledge constraints strictly. Default: False",
    )
    parser.add_argument(
        "--ci_tests",
        type=str,
        nargs="+",
        default=["pearsonr", "chi_square"],
        help="CI test(s) to use. Default: pearsonr chi_square",
    )
    return parser.parse_args()


# -------------------------------------------------------
## Main
# -------------------------------------------------------
def main():
    args = parse_args()

    significance_levels      = args.significance_level
    max_cond_vars            = args.max_cond_vars
    ci_tests                 = args.ci_tests
    enforce_expert_knowledge = args.enforce_expert_knowledge

    # Load optional expert knowledge from a pickle file
    expert_knowledge = None
    if args.expert_knowledge is not None:
        ek_path = os.path.normpath(args.expert_knowledge)
        with open(ek_path, "rb") as fh:
            expert_knowledge = pickle.load(fh)
        print(f"[INFO] Loaded expert knowledge from: {ek_path}")

    # ── Data paths ──────────────────────────────────────
    asia_datafile = os.path.normpath(
        os.path.join(
            os.getcwd(), "..", "..", "..", "Assets", "Asia", "asia_dataset.pkl"
        )
    )

    # ── Load model & data ────────────────────────────────
    asia_model, asia_data = loadData(asia_datafile)
    print(asia_model)
    print(asia_data.head())

    # ── Output directory: CadisBenchmark/Assets/Asia/<timestamp>/ ──
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.normpath(
        os.path.join(
            os.getcwd(), "..", "..", "..", "Assets", "Asia", run_timestamp
        )
    )
    os.makedirs(output_root, exist_ok=True)
    print(f"[INFO] Output directory: {output_root}")

    # ── Memory tracking: start ───────────────────────────
    tracemalloc.start()
    process = psutil.Process(os.getpid())

    # ── Runtime tracking: start ──────────────────────────
    wall_start = time.perf_counter()

    records = []

    for alpha, ci_test in product(significance_levels, ci_tests):
        print(f"  PC | alpha={alpha} | ci={ci_test} ...", end=" ", flush=True)
        try:
            pc = PC(
                ci_test=ci_test,
                significance_level=alpha,
                max_cond_vars=max_cond_vars,
                expert_knowledge=expert_knowledge,
                enforce_expert_knowledge=enforce_expert_knowledge,
                n_jobs=-1,
                show_progress=True,
            )
            pc.fit(asia_data)
            dag = pc.causal_graph_

            metrics = evaluate_structure(dag, asia_model, asia_data, "bic")
            metrics.update({
                "algorithm":               "PC",
                "alpha":                   alpha,
                "ci_test":                 ci_test,
                "max_cond_vars":           max_cond_vars,
                "expert_knowledge":        args.expert_knowledge,
                "enforce_expert_knowledge": enforce_expert_knowledge,
            })
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")

            # ── Save this run's DAG as pickle ──────────────
            safe_ci  = ci_test.replace("/", "_")
            run_label = f"PC_alpha{alpha}_ci{safe_ci}"
            dag_path  = os.path.join(output_root, f"{run_label}_dag.pkl")
            with open(dag_path, "wb") as fh:
                pickle.dump(dag, fh)
            print(f"    [SAVED] DAG → {dag_path}")

        except Exception as e:
            print(f"FAILED — {e}")
            records.append({
                "algorithm": "PC",
                "alpha":     alpha,
                "ci_test":   ci_test,
                "error":     str(e),
            })

    # ── Runtime tracking: stop ───────────────────────────
    wall_elapsed = time.perf_counter() - wall_start

    # ── Memory tracking: stop ────────────────────────────
    _current_mem, peak_mem_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_mb = process.memory_info().rss / (1024 ** 2)

    print("\n" + "=" * 60)
    print(f"  Wall-clock runtime            : {wall_elapsed:.3f} s")
    print(f"  Peak memory (tracemalloc)     : {peak_mem_tracemalloc / (1024**2):.3f} MB")
    print(f"  Process RSS (psutil)          : {rss_mb:.3f} MB")
    print("=" * 60)

    # ── Save all records + performance summary as pickle ─
    summary = {
        "records":                       records,
        "wall_elapsed_seconds":          wall_elapsed,
        "peak_memory_tracemalloc_mb":    peak_mem_tracemalloc / (1024 ** 2),
        "rss_mb":                        rss_mb,
        "run_timestamp":                 run_timestamp,
        "params": {
            "significance_levels":        significance_levels,
            "ci_tests":                   ci_tests,
            "max_cond_vars":              max_cond_vars,
            "expert_knowledge":           args.expert_knowledge,
            "enforce_expert_knowledge":   enforce_expert_knowledge,
        },
    }
    summary_path = os.path.join(output_root, "run_summary.pkl")
    with open(summary_path, "wb") as fh:
        pickle.dump(summary, fh)
    print(f"[SAVED] Full run summary → {summary_path}")

    return records


# ── Re-import helper ──────────────────────────────────────────────────────────
def load_run(output_dir: str) -> dict:
    """
    Re-import a saved run from its timestamped output directory.

    Usage
    -----
    >>> from pc import load_run
    >>> summary = load_run("path/to/Assets/Asia/20260414_153000")
    >>> print(summary["records"])
    >>> print(summary["wall_elapsed_seconds"])
    >>> dag = summary["dags"]["PC_alpha0.01_cipearsonr_dag.pkl"]
    """
    summary_path = os.path.join(output_dir, "run_summary.pkl")
    with open(summary_path, "rb") as fh:
        summary = pickle.load(fh)

    # Load individual DAG files found in the directory
    dags = {}
    for fname in os.listdir(output_dir):
        if fname.endswith("_dag.pkl"):
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "rb") as fh:
                dags[fname] = pickle.load(fh)
    summary["dags"] = dags
    return summary


if __name__ == "__main__":
    main()
