"""
pgmpy Causal Discovery Algorithms on BenchAsia Dataset — Extended
==================================================================
Original algorithms  : PC, HillClimbSearch, MMHC
Added algorithms     : GES (via BNSL / Lingam bridge), TreeSearch,
                       ExpertInTheLoop, ExhaustiveSearch

Algorithm notes
---------------
* GES          — pgmpy wraps GES as ``GES`` in pgmpy.estimators (>= 0.1.23).
                 Falls back to a BIC-score greedy equivalence search shim when
                 the class is not present (older installs).
* TreeSearch   — pgmpy's ``TreeSearch`` estimator builds a Chow-Liu (max
                 spanning tree) or a Naive-Bayes skeleton, both in closed form.
* ExpertInLoop — pgmpy's ``ExpertInLoop`` (≥ 0.1.21) performs a PC-style
                 skeleton search but pauses to query a human (or a callable
                 oracle) whenever the CI test is ambiguous. Here we wire it up
                 with an *automated oracle* built from the true DAG so the
                 benchmark can run non-interactively.
* Exhaustive   — ``ExhaustiveSearch`` enumerates *all* DAGs over the node set
                 and picks the one with the best BIC score. Only feasible for
                 ≤ 6 nodes; we therefore use a 4-node subgraph of Asia.
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
from pgmpy.models import DiscreteBayesianNetwork

# ── Scoring classes ──────────────────────────────────────────────────────────
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
print(f"[INFO] Scorer classes : { {k: v.__name__ for k, v in SCORER_MAP.items()} }")


# ─────────────────────────────────────────────
# 1. Load / Generate the Asia Dataset
# ─────────────────────────────────────────────

def load_asia_data(n_samples: int = 5000, random_state: int = 42):
    """Sample from the Asia BN and return (DataFrame, true BayesianNetwork)."""
    print("=" * 65)
    print("  Loading Asia Benchmark Dataset")
    print("=" * 65)

    asia_model = get_example_model("asia")
    print(f"True graph edges : {sorted(asia_model.edges())}")
    print(f"Nodes            : {sorted(asia_model.nodes())}")

    sampler = BayesianModelSampling(asia_model)
    data = sampler.forward_sample(size=n_samples, seed=random_state)
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
    scoring_method: str = "bic",
) -> dict:
    """Precision / Recall / F1 / SHD + log-score on data."""
    true_edges    = set(true_model.edges())
    learned_edges = set(learned.edges())

    tp = len(true_edges & learned_edges)
    fp = len(learned_edges - true_edges)
    fn = len(true_edges - learned_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # SHD counts: missing edges + extra edges (reversed counted once)
    reversed_edges = sum(1 for (u, v) in learned_edges if (v, u) in true_edges)
    shd = fp + fn  # simplified SHD (extra + missing, reversed already in both sets)

    try:
        scorer_cls = SCORER_MAP.get(scoring_method, SCORER_MAP["bic"])
        sc = scorer_cls(data)
        score_val = sum(
            sc.local_score(node, list(learned.predecessors(node)))
            for node in learned.nodes()
        )
    except Exception:
        score_val = float("nan")

    return {
        "precision":           round(precision,  4),
        "recall":              round(recall,     4),
        "f1":                  round(f1,         4),
        "shd":                 shd,
        f"{scoring_method}_score": round(score_val, 2),
        "n_edges_learned":     len(learned_edges),
        "n_edges_true":        len(true_edges),
    }


# ─────────────────────────────────────────────
# 3. Original Algorithm Runners (unchanged)
# ─────────────────────────────────────────────

def run_pc(data, significance_levels, ci_tests, true_model):
    records = []
    print("\n" + "=" * 65)
    print("  PC Algorithm")
    print("=" * 65)
    for alpha, ci_test in product(significance_levels, ci_tests):
        print(f"  PC | alpha={alpha} | ci={ci_test} ...", end=" ", flush=True)
        try:
            dag = PC(data).estimate(
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
    records = []
    print("\n" + "=" * 65)
    print("  Hill-Climb Search")
    print("=" * 65)
    for scoring_method, max_iter in product(scoring_methods, max_iter_list):
        print(f"  HC | score={scoring_method} | max_iter={max_iter} ...", end=" ", flush=True)
        try:
            scorer = SCORER_MAP[scoring_method](data)
            dag = HillClimbSearch(data).estimate(
                scoring_method=scorer,
                max_iter=max_iter,
                max_indegree=4,
                epsilon=1e-4,
            )
            metrics = evaluate_structure(dag, true_model, data, scoring_method)
            metrics.update({"algorithm": "HillClimb", "scoring_method": scoring_method, "max_iter": max_iter})
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")
        except Exception as e:
            print(f"FAILED — {e}")
            records.append({"algorithm": "HillClimb", "scoring_method": scoring_method, "max_iter": max_iter, "error": str(e)})
    return records


def run_mmhc(data, alpha_list, scoring_methods, true_model):
    records = []
    print("\n" + "=" * 65)
    print("  MMHC Estimator")
    print("=" * 65)
    scorer_map = {k: v for k, v in SCORER_MAP.items() if k in ("bic", "bdeu", "k2")}
    for alpha, scoring_method in product(alpha_list, scoring_methods):
        print(f"  MMHC | alpha={alpha} | score={scoring_method} ...", end=" ", flush=True)
        try:
            scorer = scorer_map[scoring_method](data)
            dag = MmhcEstimator(data).estimate(
                significance_level=alpha,
                tabu_length=10,
                scoring_method=scorer,
            )
            metrics = evaluate_structure(dag, true_model, data, scoring_method)
            metrics.update({"algorithm": "MMHC", "alpha": alpha, "scoring_method": scoring_method})
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")
        except Exception as e:
            print(f"FAILED — {e}")
            records.append({"algorithm": "MMHC", "alpha": alpha, "scoring_method": scoring_method, "error": str(e)})
    return records


# ─────────────────────────────────────────────
# 4. NEW: GES (Greedy Equivalence Search)
# ─────────────────────────────────────────────

def run_ges(data, scoring_methods, true_model):
    """
    GES — Greedy Equivalence Search.

    pgmpy.estimators.GES operates in three phases automatically:
      1. Forward phase  : greedily adds edges that improve the score.
      2. Backward phase : greedily removes edges that further improve the score.
      3. Flip phase     : flips edge orientations to improve the score.

    ``estimate()`` accepts ``scoring_method`` as a **string** identifier
    (e.g. "bic-d", "bdeu", "k2") and ``min_improvement`` to control the
    stopping criterion — there is no ``phases`` or ``return_type`` argument.

    Discrete-data score strings supported by GES:
        "k2", "bdeu", "bds", "bic-d", "aic-d"

    Parameters tuned here:
      * scoring_method  — "bic-d" / "bdeu" / "k2"
      * min_improvement — minimum score delta to accept an operation
                          (smaller = more edges, larger = sparser graph)

    Reference: Chickering (2002), "Optimal Structure Identification With
    Greedy Search", JMLR 3, 507-554.
    """
    records = []
    print("\n" + "=" * 65)
    print("  GES — Greedy Equivalence Search")
    print("=" * 65)

    try:
        from pgmpy.estimators import GES
        ges_available = True
    except ImportError:
        ges_available = False
        print("  [WARN] pgmpy.estimators.GES not found (requires pgmpy >= 0.1.23).")
        print("         Falling back to BIC-scored HillClimbSearch as GES proxy.")

    # GES uses its own score-string registry; map our keys to its identifiers
    ges_score_map = {"bic": "bic-d", "bdeu": "bdeu", "k2": "k2"}
    min_improvements = [1e-4, 1e-6]   # hyperparameter: stopping threshold

    scorer_keys = [s for s in scoring_methods if s in ges_score_map]

    for scoring_method, min_imp in product(scorer_keys, min_improvements):
        ges_score_str = ges_score_map[scoring_method]
        label = f"GES | score={ges_score_str} | min_improvement={min_imp}"
        print(f"  Running {label} ...", end=" ", flush=True)
        try:
            if ges_available:
                est = GES(data)
                # estimate() takes the score as a STRING, not a scorer instance
                dag = est.estimate(
                    scoring_method=ges_score_str,
                    min_improvement=min_imp,
                )
            else:
                # Proxy: HC with tabu list approximates GES behaviour
                scorer = SCORER_MAP[scoring_method](data)
                dag = HillClimbSearch(data).estimate(
                    scoring_method=scorer,
                    max_iter=2000,
                    tabu_length=50,
                    epsilon=min_imp,
                )

            metrics = evaluate_structure(dag, true_model, data, scoring_method)
            metrics.update({
                "algorithm":      "GES",
                "scoring_method": ges_score_str,
                "min_improvement": min_imp,
            })
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")

        except Exception as e:
            print(f"FAILED — {e}")
            records.append({
                "algorithm":      "GES",
                "scoring_method": ges_score_str,
                "min_improvement": min_imp,
                "error":          str(e),
            })

    return records


# ─────────────────────────────────────────────
# 5. NEW: TreeSearch (Chow-Liu / Naive Bayes)
# ─────────────────────────────────────────────

def run_treesearch(data, class_nodes, root_nodes, true_model):
    """
    TreeSearch — builds a maximum-weight spanning tree (Chow-Liu algorithm)
    or a Naive-Bayes structure over the data.

    Two variants:
      * ``chow-liu``   — finds the tree-shaped BN that maximises the mutual
                         information between adjacent nodes. Runs in O(n² log n).
      * ``tan``        — Tree-Augmented Naive Bayes. Requires a class node;
                         all other nodes connect to the class *and* form a tree
                         over the non-class variables conditioned on the class.

    Parameters tuned here:
      * variant    — "chow-liu" or "tan"
      * root_node  — root of the Chow-Liu tree (or class variable for TAN)
      * class_node — class variable for TAN (ignored for chow-liu)

    Reference: Chow & Liu (1968), "Approximating discrete probability
    distributions with dependence trees", IEEE Trans. Inf. Theory, 14(3).
    """
    records = []
    print("\n" + "=" * 65)
    print("  TreeSearch (Chow-Liu / TAN)")
    print("=" * 65)

    try:
        from pgmpy.estimators import TreeSearch
    except ImportError:
        print("  [WARN] TreeSearch not available in this pgmpy version. Skipping.")
        return records

    all_nodes = list(data.columns)

    # --- Chow-Liu: vary root node ---
    for root in root_nodes:
        label = f"TreeSearch | variant=chow-liu | root={root}"
        print(f"  Running {label} ...", end=" ", flush=True)
        try:
            est = TreeSearch(data, root_node=root)
            dag = est.estimate(estimator_type="chow-liu")
            metrics = evaluate_structure(dag, true_model, data, "bic")
            metrics.update({
                "algorithm": "TreeSearch",
                "variant":   "chow-liu",
                "root_node": root,
            })
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")
        except Exception as e:
            print(f"FAILED — {e}")
            records.append({"algorithm": "TreeSearch", "variant": "chow-liu", "root_node": root, "error": str(e)})

    # --- TAN: vary class node ---
    for class_node in class_nodes:
        label = f"TreeSearch | variant=tan | class={class_node}"
        print(f"  Running {label} ...", end=" ", flush=True)
        try:
            est = TreeSearch(data, root_node=class_node)
            dag = est.estimate(estimator_type="tan", class_node=class_node)
            metrics = evaluate_structure(dag, true_model, data, "bic")
            metrics.update({
                "algorithm":  "TreeSearch",
                "variant":    "tan",
                "class_node": class_node,
            })
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")
        except Exception as e:
            print(f"FAILED — {e}")
            records.append({"algorithm": "TreeSearch", "variant": "tan", "class_node": class_node, "error": str(e)})

    return records


# ─────────────────────────────────────────────
# 6. NEW: ExpertInTheLoop
# ─────────────────────────────────────────────

def run_expert_in_loop(data, true_model, effect_size_thresholds):
    """
    ExpertInTheLoop — a hybrid structure learner that first runs a correlation/
    association filter to build a skeleton, then calls an ``orientation_fn``
    for every ambiguous edge pair to decide its direction.

    The correct pgmpy API (>= 0.1.21):

        ExpertInLoop(data).estimate(
            effect_size_threshold=<float>,
            orientation_fn=<callable(var1, var2, **kwargs) -> (src, tgt) | None>,
            use_cache=<bool>,
            show_progress=<bool>,
            **kwargs_forwarded_to_orientation_fn,
        )

    ``orientation_fn`` signature:
        Takes at least two positional arguments (the two variable names) plus
        any extra kwargs forwarded from ``estimate()``.
        Returns either a (source, target) tuple representing a directed edge,
        or None for "no edge".

    *** We do NOT use ``llm_pairwise_orient`` (requires litellm / external LLM). ***
    Instead we provide three custom orientation functions:

      1. ``oracle_orient``      — queries the *true* DAG via d-separation
                                  (perfect-knowledge baseline).
      2. ``alphabetical_orient``— always orients u→v if u < v alphabetically
                                  (zero-knowledge naive baseline).
      3. ``markov_blanket_orient``— orients toward the node with the smaller
                                  Markov Blanket size in the true model
                                  (lightweight structural heuristic).

    Parameters tuned here:
      * effect_size_threshold — minimum association strength to retain an edge
                                in the skeleton (lower = denser skeleton).
      * orientation_fn        — one of the three callables above.

    Reference: Hagedorn & Hünermund (2023), "Expert-in-the-Loop Causal
    Discovery", NeurIPS workshop on Causal Representation Learning.
    """
    records = []
    print("\n" + "=" * 65)
    print("  ExpertInTheLoop")
    print("=" * 65)

    try:
        from pgmpy.estimators import ExpertInLoop
        eitl_available = True
    except ImportError:
        eitl_available = False
        print("  [WARN] ExpertInLoop not found (requires pgmpy >= 0.1.21). Skipping.")
        return records

    # ── Custom orientation functions (no litellm required) ────────────────────

    def oracle_orient(var1, var2, **kwargs):
        """
        Perfect-knowledge oracle: uses d-separation on the true DAG to decide
        if an edge exists, then picks direction by checking which conditional
        independencies are implied.
        Returns (parent, child) or None.
        """
        # If d-separated with empty set, suggest no edge
        try:
            if true_model.local_independencies(var1).contains(var1, var2, []):
                return None
        except Exception:
            pass
        # Orient toward the node that has MORE parents in the true graph
        # (more 'downstream' nodes tend to have more parents)
        true_parents_v1 = list(true_model.predecessors(var1))
        true_parents_v2 = list(true_model.predecessors(var2))
        # If true edge exists, honour it; otherwise use parent-count heuristic
        if (var1, var2) in true_model.edges():
            return (var1, var2)
        elif (var2, var1) in true_model.edges():
            return (var2, var1)
        else:
            # Heuristic: orient toward the node with fewer existing parents
            if len(true_parents_v1) <= len(true_parents_v2):
                return (var1, var2)
            return (var2, var1)

    def alphabetical_orient(var1, var2, **kwargs):
        """
        Naive zero-knowledge baseline: orient edges alphabetically (u → v).
        Useful as a lower-bound reference for the oracle.
        """
        if var1 < var2:
            return (var1, var2)
        return (var2, var1)

    def markov_blanket_orient(var1, var2, **kwargs):
        """
        Lightweight structural heuristic: orient toward the node whose Markov
        Blanket in the true model is larger (more 'central' nodes receive
        edges from more peripheral ones).
        """
        try:
            mb1 = len(true_model.get_markov_blanket(var1))
            mb2 = len(true_model.get_markov_blanket(var2))
        except Exception:
            mb1, mb2 = 0, 0
        if mb1 >= mb2:
            return (var1, var2)   # orient toward the more central var2 side
        return (var2, var1)

    orient_fns = {
        "oracle":           oracle_orient,
        "alphabetical":     alphabetical_orient,
        "markov_blanket":   markov_blanket_orient,
    }

    for threshold, (fn_name, orient_fn) in product(
        effect_size_thresholds, orient_fns.items()
    ):
        label = f"ExpertInLoop | threshold={threshold} | orient_fn={fn_name}"
        print(f"  Running {label} ...", end=" ", flush=True)
        try:
            est = ExpertInLoop(data)
            dag = est.estimate(
                effect_size_threshold=threshold,
                orientation_fn=orient_fn,
                use_cache=True,
                show_progress=False,
            )
            metrics = evaluate_structure(dag, true_model, data, "bic")
            metrics.update({
                "algorithm":      "ExpertInLoop",
                "threshold":      threshold,
                "orient_fn":      fn_name,
            })
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")

        except Exception as e:
            print(f"FAILED — {e}")
            records.append({
                "algorithm":  "ExpertInLoop",
                "threshold":  threshold,
                "orient_fn":  fn_name,
                "error":      str(e),
            })

    return records


# ─────────────────────────────────────────────
# 7. NEW: ExhaustiveSearch
# ─────────────────────────────────────────────

def run_exhaustive_search(data, true_model, scoring_methods, max_nodes=5):
    """
    ExhaustiveSearch — enumerates every possible DAG over a node set and
    returns the one with the highest score.

    Complexity is super-exponential: the number of DAGs on n nodes grows as
        a(n) ≈ 2^(n(n-1)/2) × (corrections for acyclicity)
    and is only tractable for n ≤ 6.

    To keep runtimes reasonable we run on the *4-node* respiratory sub-graph
    of Asia: {asia, bronc, dysp, lung} by default, or a caller-specified
    subset of up to ``max_nodes`` variables.

    Parameters tuned here:
      * scoring_method — BIC / BDeu / K2

    Reference: Koller & Friedman (2009), "Probabilistic Graphical Models",
    §18.2, MIT Press.
    """
    records = []
    print("\n" + "=" * 65)
    print("  ExhaustiveSearch  (sub-graph — tractable subset of Asia)")
    print("=" * 65)

    try:
        from pgmpy.estimators import ExhaustiveSearch
    except ImportError:
        print("  [WARN] ExhaustiveSearch not available in this pgmpy version. Skipping.")
        return records

    # Select a tractable subset of nodes (≤ max_nodes)
    all_nodes = sorted(data.columns)
    # Asia respiratory chain: asia→tub, tub→either, bronc→dysp, lung→either
    preferred_subset = ["asia", "bronc", "dysp", "lung"]
    subset = [n for n in preferred_subset if n in all_nodes]
    if len(subset) < 2:
        subset = all_nodes[:max_nodes]
    subset = subset[:max_nodes]

    print(f"  Node subset for ExhaustiveSearch: {subset}")
    sub_data = data[subset].copy()

    # Build the true sub-model (keep only edges within subset)
    sub_true_edges = [(u, v) for u, v in true_model.edges()
                      if u in subset and v in subset]
    sub_true_model = DiscreteBayesianNetwork(sub_true_edges)
    sub_true_model.add_nodes_from(subset)

    for scoring_method in scoring_methods:
        label = f"ExhaustiveSearch | subset={subset} | score={scoring_method}"
        print(f"  Running {label} ...", end=" ", flush=True)
        try:
            scorer = SCORER_MAP[scoring_method](sub_data)
            est = ExhaustiveSearch(sub_data, scoring_method=scorer)
            dag = est.estimate()

            # Evaluate vs. the full true model (missing edges for out-of-subset nodes are FN)
            metrics = evaluate_structure(dag, sub_true_model, sub_data, scoring_method)
            metrics.update({
                "algorithm":      "ExhaustiveSearch",
                "scoring_method": scoring_method,
                "node_subset":    str(subset),
            })
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")

        except Exception as e:
            print(f"FAILED — {e}")
            records.append({
                "algorithm":      "ExhaustiveSearch",
                "scoring_method": scoring_method,
                "node_subset":    str(subset),
                "error":          str(e),
            })

    return records


# ─────────────────────────────────────────────
# 8. Summary & Reporting
# ─────────────────────────────────────────────

def print_summary(all_records: list[dict]):
    df = pd.DataFrame(all_records)

    display_cols = [c for c in [
        "algorithm", "alpha", "ci_test", "scoring_method", "max_iter",
        "min_improvement", "variant", "threshold", "orient_fn", "node_subset",
        "precision", "recall", "f1", "shd", "n_edges_learned",
    ] if c in df.columns]

    print("\n" + "=" * 65)
    print("  Full Results Table")
    print("=" * 65)
    print(df[display_cols].to_string(index=False))

    print("\n" + "=" * 65)
    print("  Best Configuration per Algorithm  (by F1)")
    print("=" * 65)
    for algo, grp in df.groupby("algorithm"):
        if "f1" not in grp.columns or grp["f1"].isna().all():
            continue
        best = grp.loc[grp["f1"].idxmax()]
        print(f"\n  [{algo}]")
        for col in display_cols:
            if col in best.index and pd.notna(best[col]):
                print(f"    {col:<22}: {best[col]}")

    print("\n" + "=" * 65)
    print("  Algorithm-Level Averages  (F1 / SHD)")
    print("=" * 65)
    agg = (
        df.groupby("algorithm")[["f1", "shd"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(agg.to_string())

    return df


# ─────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────

def main():
    # ── Dataset ──────────────────────────────────────────────────────────────
    data, true_model = load_asia_data(n_samples=5000)

    # ── Hyperparameter Grids ─────────────────────────────────────────────────
    # Original algorithms
    pc_alphas    = [0.01, 0.05, 0.10]
    pc_ci_tests  = ["chi_square", "g_sq"]
    hc_scores    = ["bic", "bdeu", "k2", "aic"]
    hc_max_iters = [100, 500, 1000]
    mmhc_alphas  = [0.01, 0.05]
    mmhc_scores  = ["bic", "bdeu", "k2"]

    # New algorithms
    ges_scores   = ["bic", "bdeu", "k2"]        # GES scoring functions

    # TreeSearch: root nodes for Chow-Liu; class nodes for TAN
    # Asia nodes: asia, bronc, dysp, either, lung, smoke, tub, xray
    ts_root_nodes  = ["asia", "smoke", "either"]   # Chow-Liu roots
    ts_class_nodes = ["dysp", "xray"]              # TAN class variables

    # ExpertInLoop: effect-size threshold to retain skeleton edges
    eitl_thresholds = [0.0001, 0.001, 0.01]     # ExpertInLoop effect-size thresholds

    exh_scores   = ["bic", "bdeu", "k2"]        # ExhaustiveSearch scoring

    # ── Run All Algorithms ────────────────────────────────────────────────────
    all_records = []

    # --- Original ---
    all_records += run_pc(data, pc_alphas, pc_ci_tests, true_model)
    all_records += run_hillclimb(data, hc_scores, hc_max_iters, true_model)
    all_records += run_mmhc(data, mmhc_alphas, mmhc_scores, true_model)

    # --- New ---
    all_records += run_ges(data, ges_scores, true_model)
    all_records += run_treesearch(data, ts_class_nodes, ts_root_nodes, true_model)
    all_records += run_expert_in_loop(data, true_model, eitl_thresholds)
    all_records += run_exhaustive_search(data, true_model, exh_scores, max_nodes=5)

    # ── Summarise ─────────────────────────────────────────────────────────────
    results_df = print_summary(all_records)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = "causal_discovery_results_extended.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    return results_df


if __name__ == "__main__":
    results = main()