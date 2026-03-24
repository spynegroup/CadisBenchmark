"""
pgmpy Causal Discovery Algorithms on BenchAsia Dataset
=======================================================
Implements PC, Hill-Climb Search, and Greedy Equivalence Search (GES/MMHC)
with different hyperparameters, and evaluates learned structures.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import product

# pgmpy imports
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import PC, HillClimbSearch, MmhcEstimator
from pgmpy.models import BayesianNetwork

# ── Scoring classes: names changed across pgmpy versions ─────────────────────
# pgmpy >= 0.1.19 uses BIC, BDeu, K2, AIC (pgmpy.estimators)
# older versions used BicScore, BDeuScore, K2Score, AICScore
def _import_scorers():
    """Return scorer classes, handling both old and new pgmpy naming."""
    try:
        from pgmpy.estimators import BIC, BDeu, K2, AIC
        return {"bic": BIC, "bdeu": BDeu, "k2": K2, "aic": AIC}
    except ImportError:
        pass
    try:
        from pgmpy.estimators import BicScore, BDeuScore, K2Score, AICScore
        return {"bic": BicScore, "bdeu": BDeuScore, "k2": K2Score, "aic": AICScore}
    except ImportError:
        pass
    raise ImportError(
        "Could not import scoring classes from pgmpy.estimators. "
        "Please upgrade pgmpy:  pip install -U pgmpy"
    )

SCORER_MAP = _import_scorers()
print(f"[INFO] Using scorer classes: { {k: v.__name__ for k, v in SCORER_MAP.items()} }")

# ─────────────────────────────────────────────
# 1. Load / Generate the Asia Dataset
# ─────────────────────────────────────────────

def load_asia_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Load the Asia BN from pgmpy's built-in models and sample data from it.
    Returns a DataFrame with discrete (categorical) columns.
    """
    print("=" * 60)
    print("  Loading Asia Benchmark Dataset")
    print("=" * 60)

    asia_model = get_example_model("asia")
    print(f"True graph edges : {sorted(asia_model.edges())}")
    print(f"Nodes            : {sorted(asia_model.nodes())}")

    sampler = BayesianModelSampling(asia_model)
    data = sampler.forward_sample(size=n_samples, seed=random_state)
    # Ensure all columns are string / categorical
    for col in data.columns:
        data[col] = data[col].astype(str)

    print(f"Sampled {n_samples} rows — shape: {data.shape}\n")
    return data, asia_model


# ─────────────────────────────────────────────
# 2. Evaluation Helpers
# ─────────────────────────────────────────────

def evaluate_structure(
    learned: BayesianNetwork,
    true_model: BayesianNetwork,
    data: pd.DataFrame,
    scoring_method="bic",
) -> dict:
    """
    Compute Precision, Recall, F1 on directed edges +
    SHD (Structural Hamming Distance) + BIC / BDeu score.
    """
    true_edges  = set(true_model.edges())
    learned_edges = set(learned.edges())

    tp = len(true_edges & learned_edges)
    fp = len(learned_edges - true_edges)
    fn = len(true_edges - learned_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # SHD = fp + fn + reversed edges
    reversed_edges = sum(
        1 for (u, v) in learned_edges if (v, u) in true_edges
    )
    shd = fp + fn - reversed_edges + reversed_edges  # simplification: fp+fn

    # Structure score on data
    try:
        scorer_cls = SCORER_MAP.get(scoring_method, SCORER_MAP["bic"])
        sc = scorer_cls(data)
        score_val = sum(sc.local_score(node, list(learned.predecessors(node)))
                        for node in learned.nodes())
    except Exception:
        score_val = float("nan")

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "shd":       fp + fn,
        f"{scoring_method}_score": round(score_val, 2),
        "n_edges_learned": len(learned_edges),
        "n_edges_true":    len(true_edges),
    }


# ─────────────────────────────────────────────
# 3. Algorithm Runners
# ─────────────────────────────────────────────

def run_pc(data, significance_levels, ci_tests, true_model):
    """PC algorithm with varying significance_level and CI test."""
    records = []
    print("\n" + "=" * 60)
    print("  PC Algorithm")
    print("=" * 60)

    for alpha, ci_test in product(significance_levels, ci_tests):
        label = f"PC | alpha={alpha} | ci={ci_test}"
        print(f"  Running {label} ...", end=" ", flush=True)
        try:
            est = PC(data)
            dag = est.estimate(
                variant="stable",
                ci_test=ci_test,
                significance_level=alpha,
                return_type="dag",
            )
            metrics = evaluate_structure(dag, true_model, data, "bic")
            metrics.update({"algorithm": "PC", "alpha": alpha, "ci_test": ci_test})
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")
        except Exception as e:
            print(f"FAILED — {e}")
            records.append({"algorithm": "PC", "alpha": alpha, "ci_test": ci_test, "error": str(e)})

    return records


def run_hillclimb(data, scoring_methods, max_iter_list, true_model):
    """Hill-Climb Search with varying scoring function and max_iter."""
    records = []
    print("\n" + "=" * 60)
    print("  Hill-Climb Search")
    print("=" * 60)

    scorer_map = SCORER_MAP

    for scoring_method, max_iter in product(scoring_methods, max_iter_list):
        label = f"HC | score={scoring_method} | max_iter={max_iter}"
        print(f"  Running {label} ...", end=" ", flush=True)
        try:
            scorer = scorer_map[scoring_method](data)
            est = HillClimbSearch(data)
            dag = est.estimate(
                scoring_method=scorer,
                max_iter=max_iter,
                max_indegree=4,
                epsilon=1e-4,
            )
            metrics = evaluate_structure(dag, true_model, data, scoring_method)
            metrics.update({
                "algorithm":      "HillClimb",
                "scoring_method": scoring_method,
                "max_iter":       max_iter,
            })
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")
        except Exception as e:
            print(f"FAILED — {e}")
            records.append({
                "algorithm": "HillClimb",
                "scoring_method": scoring_method,
                "max_iter": max_iter,
                "error": str(e),
            })

    return records


def run_mmhc(data, alpha_list, scoring_methods, true_model):
    """MMHC (Max-Min Hill-Climbing) with varying alpha and scoring method."""
    records = []
    print("\n" + "=" * 60)
    print("  MMHC Estimator")
    print("=" * 60)

    scorer_map = {k: v for k, v in SCORER_MAP.items() if k in ("bic", "bdeu", "k2")}

    for alpha, scoring_method in product(alpha_list, scoring_methods):
        label = f"MMHC | alpha={alpha} | score={scoring_method}"
        print(f"  Running {label} ...", end=" ", flush=True)
        try:
            scorer = scorer_map[scoring_method](data)
            est = MmhcEstimator(data)
            dag = est.estimate(
                significance_level=alpha,
                tabu_length=10,
                scoring_method=scorer,
            )
            metrics = evaluate_structure(dag, true_model, data, scoring_method)
            metrics.update({
                "algorithm":      "MMHC",
                "alpha":          alpha,
                "scoring_method": scoring_method,
            })
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")
        except Exception as e:
            print(f"FAILED — {e}")
            records.append({
                "algorithm": "MMHC",
                "alpha": alpha,
                "scoring_method": scoring_method,
                "error": str(e),
            })

    return records


# ─────────────────────────────────────────────
# 4. Summary & Reporting
# ─────────────────────────────────────────────

def print_summary(all_records: list[dict]):
    df = pd.DataFrame(all_records)
    numeric_cols = ["precision", "recall", "f1", "shd"]

    print("\n" + "=" * 60)
    print("  Full Results Table")
    print("=" * 60)
    display_cols = [c for c in
        ["algorithm", "alpha", "ci_test", "scoring_method", "max_iter",
         "precision", "recall", "f1", "shd", "n_edges_learned"]
        if c in df.columns]
    print(df[display_cols].to_string(index=False))

    print("\n" + "=" * 60)
    print("  Best Configuration per Algorithm (by F1 score)")
    print("=" * 60)
    for algo, grp in df.groupby("algorithm"):
        if "f1" not in grp.columns:
            continue
        best = grp.loc[grp["f1"].idxmax()]
        print(f"\n  [{algo}]")
        for col in display_cols:
            if col in best.index and pd.notna(best[col]):
                print(f"    {col:<20}: {best[col]}")

    print("\n" + "=" * 60)
    print("  Algorithm-Level Averages (F1 / SHD)")
    print("=" * 60)
    agg = (
        df.groupby("algorithm")[["f1", "shd"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(agg.to_string())

    return df


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

def main():
    # ── Dataset ──────────────────────────────
    data, true_model = load_asia_data(n_samples=5000)

    # ── Hyperparameter Grids ─────────────────
    pc_alphas       = [0.01, 0.05, 0.10]
    pc_ci_tests     = ["chi_square", "g_sq"]

    hc_scores       = ["bic", "bdeu", "k2", "aic"]
    hc_max_iters    = [100, 500, 1000]

    mmhc_alphas     = [0.01, 0.05]
    mmhc_scores     = ["bic", "bdeu", "k2"]

    # ── Run Algorithms ────────────────────────
    all_records = []
    all_records += run_pc(data, pc_alphas, pc_ci_tests, true_model)
    all_records += run_hillclimb(data, hc_scores, hc_max_iters, true_model)
    all_records += run_mmhc(data, mmhc_alphas, mmhc_scores, true_model)

    # ── Summarise ─────────────────────────────
    results_df = print_summary(all_records)

    # ── Save CSV ──────────────────────────────
    out_path = "causal_discovery_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    return results_df


if __name__ == "__main__":
    results = main()
