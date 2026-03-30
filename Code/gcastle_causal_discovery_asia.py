"""
gCastle Causal Discovery — Full Algorithm Benchmark on Asia Dataset
====================================================================
Runs every IID-compatible algorithm available in gCastle (v1.0.3+) on the
Asia Bayesian Network benchmark dataset.

Algorithm inventory (13 algorithms across 3 families)
------------------------------------------------------
Constraint-based
  1.  PC                — Peter-Clark, conditional independence tests
Score-based / classical
  2.  GES               — Greedy Equivalence Search
Functional / LiNGAM family
  3.  ICALiNGAM         — ICA-based LiNGAM
  4.  DirectLiNGAM      — Direct (regression-based) LiNGAM
  5.  ANMNonlinear      — Additive Noise Model (nonlinear, HSIC test)
Gradient-based (continuous optimisation)
  6.  Notears           — Linear NOTEARS (L1/L2 penalty)
  7.  NotearsNonlinear  — Nonlinear NOTEARS (MLP / Sobolev)
  8.  GOLEM             — Gradient-based Optimised Likelihood EM
  9.  GraNDAG           — Gradient-based Neural DAG Learning
  10. MCSL              — Masked Causal Structure Learning
  11. GAE               — Graph Auto-Encoder for causal structure
Reinforcement-learning-based
  12. RL                — Reinforcement Learning (flexible score)
  13. CORL              — Combinatorial RL (order-based)

Asia dataset notes
------------------
The Asia BN is a discrete (binary) network with 8 nodes and 8 directed edges.
gCastle algorithms that assume *continuous* data (all gradient-based and
LiNGAM methods) receive ordinal-encoded float data (0.0 / 1.0) instead of raw
category strings.  The constraint-based PC and score-based GES algorithms
receive the same float matrix; PC uses the chi2 CI test to remain appropriate
for discrete data.

Dependencies
------------
  pip install gcastle pgmpy        # core
  pip install torch                # required by gradient-based algorithms
  pip install scikit-learn         # required by ANMNonlinear (GPR)

Usage
-----
  python gcastle_causal_discovery_asia.py

  Results are printed to stdout and saved as:
    gcastle_asia_results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd

# ── pgmpy: used only to generate / sample the Asia dataset ───────────────────
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling

# ── gCastle core ─────────────────────────────────────────────────────────────
from castle.metrics import MetricsDAG

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Asia dataset loader
# ─────────────────────────────────────────────────────────────────────────────

def load_asia(n_samples: int = 5000, seed: int = 42):
    """
    Sample from the Asia Bayesian Network (pgmpy built-in).

    Returns
    -------
    X_float : np.ndarray, shape (n_samples, 8)
        Ordinal-encoded float matrix (0.0 / 1.0).  Suitable for all gCastle
        algorithms that require continuous input.
    true_adj : np.ndarray, shape (8, 8)
        Binary adjacency matrix of the true DAG (true_adj[i,j]=1 ↔ i→j).
    node_names : list[str]
        Variable names in column order.
    """
    print("=" * 68)
    print("  Asia Benchmark Dataset")
    print("=" * 68)

    asia = get_example_model("asia")
    sampler = BayesianModelSampling(asia)
    df = sampler.forward_sample(size=n_samples, seed=seed)

    # Encode each binary categorical to 0 / 1 float
    node_names = sorted(df.columns)          # alphabetical, consistent ordering
    df = df[node_names]
    for col in node_names:
        df[col] = pd.Categorical(df[col]).codes.astype(float)

    X_float = df.values                       # (n_samples, 8)

    # Build true adjacency matrix in the same column order
    n = len(node_names)
    idx = {v: i for i, v in enumerate(node_names)}
    true_adj = np.zeros((n, n), dtype=int)
    for u, v in asia.edges():
        true_adj[idx[u], idx[v]] = 1

    print(f"Nodes  : {node_names}")
    print(f"Edges  : {list(asia.edges())}")
    print(f"Samples: {n_samples}  |  shape: {X_float.shape}")
    print(f"True adjacency matrix:\n{true_adj}\n")
    return X_float, true_adj, node_names


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(causal_matrix: np.ndarray, true_adj: np.ndarray, algo: str,
             elapsed: float, extra: dict = None) -> dict:
    """
    Compute MetricsDAG (F1, precision, recall, SHD, FDR, TPR, FPR, NNZ) and
    package into a result record.

    gCastle's MetricsDAG treats weighted matrices correctly: any non-zero entry
    is interpreted as an edge.  We binarise before passing to be safe.
    """
    pred = (causal_matrix != 0).astype(int)
    try:
        m = MetricsDAG(pred, true_adj).metrics
    except Exception as e:
        m = {"error_metric": str(e)}

    record = {
        "algorithm":  algo,
        "elapsed_s":  round(elapsed, 2),
        "f1":         round(m.get("F1",  float("nan")), 4),
        "precision":  round(m.get("precision",  float("nan")), 4),
        "recall":     round(m.get("recall",     float("nan")), 4),
        "shd":        m.get("SHD",  float("nan")),
        "fdr":        round(m.get("FDR", float("nan")), 4),
        "tpr":        round(m.get("TPR", float("nan")), 4),
        "fpr":        round(m.get("FPR", float("nan")), 4),
        "nnz":        m.get("nnz",  float("nan")),
        "n_edges_true": int(true_adj.sum()),
    }
    if extra:
        record.update(extra)
    return record


def _run(name, fn, true_adj, extra=None):
    """Wrapper: time and catch exceptions around a learn-and-return call."""
    print(f"  [{name}] Running ...", end=" ", flush=True)
    t0 = time.time()
    try:
        mat = fn()
        elapsed = time.time() - t0
        rec = evaluate(mat, true_adj, name, elapsed, extra)
        print(f"done ({elapsed:.1f}s) | F1={rec['f1']:.3f}  SHD={rec['shd']}")
    except Exception as e:
        elapsed = time.time() - t0
        rec = {"algorithm": name, "elapsed_s": round(elapsed, 2),
               "error": str(e)}
        print(f"FAILED ({elapsed:.1f}s) — {e}")
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Algorithm runners — one function per algorithm family
# ─────────────────────────────────────────────────────────────────────────────

# ── 3.1  PC (Constraint-based) ───────────────────────────────────────────────

def run_pc(X, true_adj):
    """
    PC Algorithm — Peter-Clark
    --------------------------
    Learns a CPDAG via conditional independence tests and Meek orientation rules.

    API:
        PC(variant, alpha, ci_test, priori_knowledge)

    Hyperparameters swept:
        variant  — 'original' | 'stable' | 'parallel'
                   'stable' fixes the ordering issue in the original PC.
        alpha    — significance level for CI tests (0.01, 0.05, 0.10)
        ci_test  — 'fisherz' (Gaussian data), 'g2' (discrete log-likelihood),
                   'chi2' (discrete chi-squared)
                   We use 'chi2' — most appropriate for binary discrete data.

    Reference: Spirtes et al. (2000), Causation, Prediction, and Search.
    """
    from castle.algorithms import PC

    records = []
    print("\n" + "=" * 68)
    print("  1. PC — Peter-Clark Algorithm")
    print("=" * 68)

    for variant in ["original", "stable"]:
        for alpha in [0.01, 0.05, 0.10]:
            name = f"PC|variant={variant}|alpha={alpha}"
            extra = {"variant": variant, "alpha": alpha, "ci_test": "chi2"}
            records.append(_run(
                name,
                lambda v=variant, a=alpha: PC(variant=v, alpha=a, ci_test="chi2")
                    .learn(X) or PC(variant=v, alpha=a, ci_test="chi2")
                    .causal_matrix,
                true_adj, extra
            ))

    # Cleaner version that avoids double-init
    records2 = []
    for variant in ["original", "stable"]:
        for alpha in [0.01, 0.05, 0.10]:
            name = f"PC|variant={variant}|alpha={alpha}"
            extra = {"variant": variant, "alpha": alpha, "ci_test": "chi2"}

            def _learn_pc(v=variant, a=alpha):
                m = PC(variant=v, alpha=a, ci_test="chi2")
                m.learn(X)
                return m.causal_matrix

            records2.append(_run(name, _learn_pc, true_adj, extra))

    return records2


# ── 3.2  GES (Score-based) ───────────────────────────────────────────────────

def run_ges(X, true_adj):
    """
    GES — Greedy Equivalence Search
    --------------------------------
    Score-based search over the space of Markov Equivalence Classes.
    Three phases: forward (add edges), backward (remove edges), flip (orient).

    API:
        GES(criterion, method, k, N)

    Hyperparameters swept:
        criterion — 'bic' | 'bdeu'  (BIC for Gaussian, BDeu for discrete)
        method    — 'scatter' | 'r2'  (covariance estimation method)
                    'scatter' is the standard; 'r2' uses R² for scoring.

    Reference: Chickering (2002), JMLR 3:507-554.
    """
    from castle.algorithms import GES

    records = []
    print("\n" + "=" * 68)
    print("  2. GES — Greedy Equivalence Search")
    print("=" * 68)

    for criterion in ["bic", "bdeu"]:
        for method in ["scatter", "r2"]:
            name = f"GES|criterion={criterion}|method={method}"
            extra = {"criterion": criterion, "method": method}

            def _learn_ges(c=criterion, meth=method):
                m = GES(criterion=c, method=meth)
                m.learn(X)
                return m.causal_matrix

            records.append(_run(name, _learn_ges, true_adj, extra))

    return records


# ── 3.3  ICALiNGAM ───────────────────────────────────────────────────────────

def run_icalingam(X, true_adj):
    """
    ICALiNGAM — ICA-based Linear Non-Gaussian Acyclic Model
    --------------------------------------------------------
    Uses FastICA to recover the mixing matrix and infer causal order.
    Assumes linear SEMs with non-Gaussian noise.

    API:
        ICALiNGAM(random_state, max_iter, thresh)

    Hyperparameters swept:
        max_iter — maximum FastICA iterations (500, 1000, 2000)
        thresh   — threshold for pruning near-zero coefficients (0.1, 0.3, 0.5)

    Note: Asia data is discrete/binary.  LiNGAM methods assume continuous
    non-Gaussian distributions.  Results serve as a methodological comparison
    (showing the penalty of assumption violation).

    Reference: Shimizu et al. (2006), JMLR 7:2003-2030.
    """
    from castle.algorithms import ICALiNGAM

    records = []
    print("\n" + "=" * 68)
    print("  3. ICALiNGAM — ICA-based LiNGAM")
    print("=" * 68)

    for max_iter in [500, 1000]:
        for thresh in [0.1, 0.3, 0.5]:
            name = f"ICALiNGAM|max_iter={max_iter}|thresh={thresh}"
            extra = {"max_iter": max_iter, "thresh": thresh}

            def _learn(mi=max_iter, th=thresh):
                m = ICALiNGAM(random_state=42, max_iter=mi, thresh=th)
                m.learn(X)
                return m.causal_matrix

            records.append(_run(name, _learn, true_adj, extra))

    return records


# ── 3.4  DirectLiNGAM ────────────────────────────────────────────────────────

def run_directlingam(X, true_adj):
    """
    DirectLiNGAM — Direct Method for Linear Non-Gaussian Acyclic Models
    --------------------------------------------------------------------
    Iteratively identifies exogenous variables via regression residual
    independence tests, avoiding ICA and its convergence issues.

    API:
        DirectLiNGAM(random_state, prior_knowledge, apply_prior_knowledge_three_rule,
                     measure)

    Hyperparameters swept:
        measure — independence measure: 'pwling' (pairwise likelihood) |
                  'kernel' (HSIC-based kernel independence test)

    Reference: Shimizu et al. (2011), JMLR 12:1225-1248.
    """
    from castle.algorithms import DirectLiNGAM

    records = []
    print("\n" + "=" * 68)
    print("  4. DirectLiNGAM — Direct LiNGAM")
    print("=" * 68)

    for measure in ["pwling", "kernel"]:
        name = f"DirectLiNGAM|measure={measure}"
        extra = {"measure": measure}

        def _learn(ms=measure):
            m = DirectLiNGAM(random_state=42, measure=ms)
            m.learn(X)
            return m.causal_matrix

        records.append(_run(name, _learn, true_adj, extra))

    return records


# ── 3.5  ANMNonlinear ────────────────────────────────────────────────────────

def run_anm(X, true_adj):
    """
    ANMNonlinear — Additive Noise Model (Nonlinear)
    ------------------------------------------------
    For each pair (Xi, Xj), fits Xi→Xj and Xj→Xi using a nonlinear regressor
    (default: Gaussian Process Regression) and uses the HSIC test to check
    residual independence.  Orientates the edge in the direction where
    residuals are independent of the input.

    API:
        ANMNonlinear(alpha)
        learn(data, regressor, test_method)

    Hyperparameters swept:
        alpha — HSIC significance level (0.01, 0.05)

    Note: Quadratic runtime O(n²) in the number of variable pairs.  For
    8-node Asia this means 28 pairwise tests — feasible but slow with GPR.

    Reference: Hoyer et al. (2009), NeurIPS.
    """
    from castle.algorithms import ANMNonlinear

    records = []
    print("\n" + "=" * 68)
    print("  5. ANMNonlinear — Additive Noise Model")
    print("=" * 68)

    for alpha in [0.01, 0.05]:
        name = f"ANMNonlinear|alpha={alpha}"
        extra = {"alpha": alpha}

        def _learn(a=alpha):
            m = ANMNonlinear(alpha=a)
            m.learn(X)
            return m.causal_matrix

        records.append(_run(name, _learn, true_adj, extra))

    return records


# ── 3.6  NOTEARS (linear) ────────────────────────────────────────────────────

def run_notears(X, true_adj):
    """
    Notears — Non-combinatorial Optimisation via Trace Exponential and
              Augmented Lagrangian for Structure learning (linear)
    ---------------------------------------------------------------
    Reformulates DAG learning as a continuous constrained optimisation problem
    using the acyclicity constraint h(W) = tr(e^(W⊙W)) - d = 0 and solves
    with augmented Lagrangian + L-BFGS-B.

    API:
        Notears(lambda1, lambda2, loss_type, max_iter, h_tol, rho_max,
                w_threshold)

    Hyperparameters swept:
        lambda1      — L1 sparsity on W (0.01, 0.1)
        loss_type    — 'l2' (linear Gaussian) | 'logistic' | 'poisson'
        w_threshold  — post-processing threshold to zero small weights (0.3, 0.5)

    Reference: Zheng et al. (2018), NeurIPS.
    """
    from castle.algorithms import Notears

    records = []
    print("\n" + "=" * 68)
    print("  6. Notears — Linear NOTEARS")
    print("=" * 68)

    for lambda1 in [0.01, 0.1]:
        for loss_type in ["l2", "logistic"]:
            for w_thresh in [0.3, 0.5]:
                name = f"Notears|lambda1={lambda1}|loss={loss_type}|thresh={w_thresh}"
                extra = {"lambda1": lambda1, "loss_type": loss_type,
                         "w_threshold": w_thresh}

                def _learn(l1=lambda1, lt=loss_type, wt=w_thresh):
                    m = Notears(lambda1=l1, loss_type=lt, w_threshold=wt)
                    m.learn(X)
                    return m.causal_matrix

                records.append(_run(name, _learn, true_adj, extra))

    return records


# ── 3.7  NotearsNonlinear ────────────────────────────────────────────────────

def run_notears_nonlinear(X, true_adj):
    """
    NotearsNonlinear — Nonlinear NOTEARS (MLP / Sobolev)
    -----------------------------------------------------
    Extends NOTEARS to nonlinear SEMs by parameterising each structural
    equation with a neural network (MLP) or a Sobolev-space basis (sob).

    API:
        NotearsNonlinear(lambda1, lambda2, max_iter, h_tol, rho_max,
                         w_threshold, model_type, device_type)

    Hyperparameters swept:
        model_type — 'mlp' (multi-layer perceptron) | 'sob' (Sobolev)
        lambda1    — L1 regularisation (0.01, 0.1)

    Reference: Lachapelle et al. (2020) / Zheng et al. (2020).
    """
    from castle.algorithms import NotearsNonlinear

    records = []
    print("\n" + "=" * 68)
    print("  7. NotearsNonlinear — Nonlinear NOTEARS (MLP/Sob)")
    print("=" * 68)

    for model_type in ["mlp", "sob"]:
        for lambda1 in [0.01, 0.1]:
            name = f"NotearsNL|model={model_type}|lambda1={lambda1}"
            extra = {"model_type": model_type, "lambda1": lambda1}

            def _learn(mt=model_type, l1=lambda1):
                m = NotearsNonlinear(
                    lambda1=l1,
                    lambda2=0.01,
                    max_iter=1000,
                    w_threshold=0.3,
                    model_type=mt,
                    device_type="cpu",
                )
                m.learn(X)
                return m.causal_matrix

            records.append(_run(name, _learn, true_adj, extra))

    return records


# ── 3.8  GOLEM ───────────────────────────────────────────────────────────────

def run_golem(X, true_adj):
    """
    GOLEM — Gradient-based Optimisation of dag-penalised Likelihood for
            learning linEar dag Models
    -------------------------------------------------------------------
    A more sample-efficient alternative to NOTEARS that minimises a
    penalised negative log-likelihood with the same acyclicity constraint
    but uses a smoother formulation, reducing the number of iterations needed.

    API:
        GOLEM(B_init, lambda_1, lambda_2, equal_variances, non_equal_variances,
              learning_rate, num_iter, checkpoint_iter, seed, graph_thres,
              device_type, device_ids)

    Hyperparameters swept:
        equal_variances — True (EV-GOLEM, assumes same noise variance)
                        | False (NV-GOLEM, unequal variances)
        lambda_1        — L1 DAG sparsity penalty (0.02, 0.05)
        lambda_2        — acyclicity penalty strength (5.0)

    Reference: Ng et al. (2020), NeurIPS.
    """
    from castle.algorithms import GOLEM

    records = []
    print("\n" + "=" * 68)
    print("  8. GOLEM — Gradient-based Optimised Likelihood EM")
    print("=" * 68)

    for equal_var in [True, False]:
        for lambda_1 in [0.02, 0.05]:
            var_label = "EV" if equal_var else "NV"
            name = f"GOLEM|{var_label}|lambda1={lambda_1}"
            extra = {"equal_variances": equal_var, "lambda_1": lambda_1}

            def _learn(ev=equal_var, l1=lambda_1):
                m = GOLEM(
                    lambda_1=l1,
                    lambda_2=5.0,
                    equal_variances=ev,
                    learning_rate=1e-3,
                    num_iter=int(1e4),      # reduced from 1e5 for speed
                    graph_thres=0.3,
                    device_type="cpu",
                    seed=42,
                )
                m.learn(X)
                return m.causal_matrix

            records.append(_run(name, _learn, true_adj, extra))

    return records


# ── 3.9  GraNDAG ─────────────────────────────────────────────────────────────

def run_grandag(X, true_adj):
    """
    GraNDAG — Gradient-based Neural DAG Learning
    ---------------------------------------------
    Models each structural equation with a neural network whose input mask
    encodes the adjacency matrix.  Jointly optimises the structural equations
    and the DAG constraint using a smooth Lagrangian penalty.

    API:
        GraNDAG(input_dim, hidden_num, hidden_dim, batch_size, lr, iterations,
                model_name, nonlinear, optimizer, h_threshold, device_type,
                use_pns, pns_thresh, num_neighbors, normalize, precision,
                random_seed, jac_thresh, lambda_init, mu_init, omega_lambda,
                omega_mu, stop_crit_win, edge_clamp_range, norm_prod,
                square_prod)

    Hyperparameters swept:
        model_name — 'NonLinGaussANM' (nonlinear Gaussian ANM) |
                     'NonLinGauss'    (nonlinear Gaussian)
        hidden_dim — neurons per hidden layer (10, 20)

    Reference: Lachapelle et al. (2020), ICLR.
    """
    from castle.algorithms import GraNDAG

    records = []
    print("\n" + "=" * 68)
    print("  9. GraNDAG — Gradient-based Neural DAG Learning")
    print("=" * 68)

    n_vars = X.shape[1]

    for model_name in ["NonLinGaussANM", "NonLinGauss"]:
        for hidden_dim in [10, 20]:
            name = f"GraNDAG|model={model_name}|hidden={hidden_dim}"
            extra = {"model_name": model_name, "hidden_dim": hidden_dim}

            def _learn(mn=model_name, hd=hidden_dim):
                m = GraNDAG(
                    input_dim=n_vars,
                    hidden_num=2,
                    hidden_dim=hd,
                    batch_size=64,
                    lr=1e-3,
                    iterations=2000,        # reduced for speed
                    model_name=mn,
                    nonlinear="leaky-relu",
                    optimizer="rmsprop",
                    h_threshold=1e-7,
                    device_type="cpu",
                    use_pns=False,
                    normalize=False,
                    random_seed=42,
                    jac_thresh=True,
                    mu_init=1e-3,
                    omega_mu=0.9,
                    stop_crit_win=100,
                    edge_clamp_range=1e-4,
                    norm_prod="paths",
                    square_prod=False,
                )
                m.learn(X)
                return m.causal_matrix

            records.append(_run(name, _learn, true_adj, extra))

    return records


# ── 3.10  MCSL ───────────────────────────────────────────────────────────────

def run_mcsl(X, true_adj):
    """
    MCSL — Masked Causal Structure Learning
    ----------------------------------------
    Learns the binary adjacency matrix directly by training a masked
    autoencoder; the binary mask is learned via straight-through estimator.
    Handles nonlinear additive noise.

    API:
        MCSL(model_type, num_hidden_layers, hidden_dim, graph_thresh,
             l1_graph_penalty, learning_rate, max_iter, iter_step,
             init_iter, h_tol, rho_thresh, w_threshold, device_type,
             use_gpu, seed)

    Hyperparameters swept:
        model_type — 'nn' (neural network) | 'qr' (quadratic regression)
        l1_graph_penalty — sparsity on the binary mask (2e-3, 1e-2)

    Reference: Ng et al. (2019).
    """
    from castle.algorithms import MCSL

    records = []
    print("\n" + "=" * 68)
    print("  10. MCSL — Masked Causal Structure Learning")
    print("=" * 68)

    for model_type in ["nn", "qr"]:
        for penalty in [2e-3, 1e-2]:
            name = f"MCSL|model={model_type}|penalty={penalty}"
            extra = {"model_type": model_type, "l1_graph_penalty": penalty}

            def _learn(mt=model_type, p=penalty):
                m = MCSL(
                    model_type=mt,
                    num_hidden_layers=2,
                    hidden_dim=16,
                    graph_thresh=0.5,
                    l1_graph_penalty=p,
                    learning_rate=3e-3,
                    max_iter=5000,
                    iter_step=50,
                    init_iter=3,
                    h_tol=1e-10,
                    rho_thresh=1e14,
                    w_threshold=0.5,
                    device_type="cpu",
                    use_gpu=False,
                    seed=42,
                )
                m.learn(X)
                return m.causal_matrix

            records.append(_run(name, _learn, true_adj, extra))

    return records


# ── 3.11  GAE ────────────────────────────────────────────────────────────────

def run_gae(X, true_adj):
    """
    GAE — Graph Auto-Encoder for Causal Structure Learning
    -------------------------------------------------------
    Encodes graph structure with a GNN encoder and decodes it via a DAG-
    constrained decoder.  Suitable for nonlinear causal relationships.

    API:
        GAE(input_dim, hidden_layers, hidden_dim, activation, epochs,
            batch_size, learning_rate, rho, alpha, beta, init_rho,
            rho_thresh, h_thresh, multiply_h, graph_thresh, seed,
            device_type)

    Hyperparameters swept:
        hidden_dim  — encoder hidden dimension (16, 32)
        graph_thresh — adjacency threshold (0.3, 0.5)

    Reference: Based on Kipf & Welling (2016) VGAE, adapted for causal learning.
    """
    from castle.algorithms import GAE

    records = []
    print("\n" + "=" * 68)
    print("  11. GAE — Graph Auto-Encoder")
    print("=" * 68)

    n_vars = X.shape[1]

    for hidden_dim in [16, 32]:
        for graph_thresh in [0.3, 0.5]:
            name = f"GAE|hidden={hidden_dim}|thresh={graph_thresh}"
            extra = {"hidden_dim": hidden_dim, "graph_thresh": graph_thresh}

            def _learn(hd=hidden_dim, gt=graph_thresh):
                m = GAE(
                    input_dim=n_vars,
                    hidden_layers=1,
                    hidden_dim=hd,
                    activation="relu",
                    epochs=200,
                    batch_size=64,
                    learning_rate=1e-3,
                    graph_thresh=gt,
                    seed=42,
                    device_type="cpu",
                )
                m.learn(X)
                return m.causal_matrix

            records.append(_run(name, _learn, true_adj, extra))

    return records


# ── 3.12  RL ─────────────────────────────────────────────────────────────────

def run_rl(X, true_adj):
    """
    RL — Reinforcement Learning for Causal Structure Learning
    ---------------------------------------------------------
    Treats each DAG as an action and trains a policy network (encoder-decoder
    Transformer) to maximise a graph quality reward (BIC, log-likelihood, etc.).
    Supports non-smooth and non-differentiable score functions.

    API:
        RL(encoder_type, input_dim, embed_dim, normalize, encoder_name,
           encoder_heads, encoder_blocks, encoder_dropout_rate, decoder_name,
           reward_mode, reward_score_type, reward_regression_type,
           reward_gpr_alpha, iteration, actor_lr, critic_lr, alpha,
           init_baseline, random_seed, device_type, device_ids)

    Hyperparameters swept:
        reward_score_type — 'BIC' | 'BIC_different_var'
        iteration         — training iterations (100, 300)

    Reference: Zhu et al. (2020), RL-BIC paper.
    """
    from castle.algorithms import RL

    records = []
    print("\n" + "=" * 68)
    print("  12. RL — Reinforcement Learning")
    print("=" * 68)

    n_vars = X.shape[1]

    for reward_type in ["BIC", "BIC_different_var"]:
        for iteration in [100, 300]:
            name = f"RL|reward={reward_type}|iter={iteration}"
            extra = {"reward_score_type": reward_type, "iteration": iteration}

            def _learn(rt=reward_type, it=iteration):
                m = RL(
                    encoder_type="TransformerEncoder",
                    input_dim=n_vars,
                    embed_dim=64,
                    normalize=False,
                    encoder_name="transformer",
                    encoder_heads=8,
                    encoder_blocks=3,
                    encoder_dropout_rate=0.1,
                    decoder_name="lstm",
                    reward_mode="episodic",
                    reward_score_type=rt,
                    reward_regression_type="LR",
                    reward_gpr_alpha=1.0,
                    iteration=it,
                    actor_lr=1e-4,
                    critic_lr=1e-3,
                    alpha=0.99,
                    init_baseline=-1.0,
                    random_seed=42,
                    device_type="cpu",
                    device_ids=0,
                )
                m.learn(X)
                return m.causal_matrix

            records.append(_run(name, _learn, true_adj, extra))

    return records


# ── 3.13  CORL ───────────────────────────────────────────────────────────────

def run_corl(X, true_adj):
    """
    CORL — Combinatorial Optimisation via Reinforcement Learning (order-based)
    --------------------------------------------------------------------------
    Improves over RL by searching over *topological orderings* rather than
    full DAGs, reducing the action space and improving scalability.
    A pointer network decoder constructs the ordering autoregressively.

    API:
        CORL(encoder_type, input_dim, embed_dim, normalize, encoder_name,
             encoder_heads, encoder_blocks, encoder_dropout_rate, decoder_name,
             reward_mode, reward_score_type, reward_regression_type,
             reward_gpr_alpha, iteration, actor_lr, critic_lr, alpha,
             init_baseline, random_seed, device_type, device_ids)

    Hyperparameters swept:
        reward_score_type — 'BIC' | 'BIC_different_var'
        iteration         — training iterations (100, 300)

    Reference: Wang et al. (2021), arXiv:2105.06631.
    """
    from castle.algorithms import CORL

    records = []
    print("\n" + "=" * 68)
    print("  13. CORL — Combinatorial RL (order-based)")
    print("=" * 68)

    n_vars = X.shape[1]

    for reward_type in ["BIC", "BIC_different_var"]:
        for iteration in [100, 300]:
            name = f"CORL|reward={reward_type}|iter={iteration}"
            extra = {"reward_score_type": reward_type, "iteration": iteration}

            def _learn(rt=reward_type, it=iteration):
                m = CORL(
                    encoder_type="TransformerEncoder",
                    input_dim=n_vars,
                    embed_dim=64,
                    normalize=False,
                    encoder_name="transformer",
                    encoder_heads=8,
                    encoder_blocks=3,
                    encoder_dropout_rate=0.1,
                    decoder_name="lstm",
                    reward_mode="episodic",
                    reward_score_type=rt,
                    reward_regression_type="LR",
                    reward_gpr_alpha=1.0,
                    iteration=it,
                    actor_lr=1e-4,
                    critic_lr=1e-3,
                    alpha=0.99,
                    init_baseline=-1.0,
                    random_seed=42,
                    device_type="cpu",
                    device_ids=0,
                )
                m.learn(X)
                return m.causal_matrix

            records.append(_run(name, _learn, true_adj, extra))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Summary & reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_records: list[dict]):
    df = pd.DataFrame(all_records)

    # Derive base algorithm name (before the first '|')
    df["algo_family"] = df["algorithm"].str.split("|").str[0]

    key_cols = ["algorithm", "f1", "precision", "recall", "shd",
                "fdr", "tpr", "elapsed_s", "n_edges_true", "nnz"]
    display_cols = [c for c in key_cols if c in df.columns]

    print("\n" + "=" * 68)
    print("  FULL RESULTS TABLE")
    print("=" * 68)
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print(df[display_cols].to_string(index=False))

    print("\n" + "=" * 68)
    print("  BEST CONFIGURATION PER ALGORITHM FAMILY  (by F1)")
    print("=" * 68)
    for family, grp in df.groupby("algo_family"):
        if "f1" not in grp.columns or grp["f1"].isna().all():
            continue
        valid = grp.dropna(subset=["f1"])
        if valid.empty:
            continue
        best = valid.loc[valid["f1"].idxmax()]
        print(f"\n  [{family}]")
        for col in display_cols:
            if col in best.index and pd.notna(best[col]):
                print(f"    {col:<22}: {best[col]}")

    print("\n" + "=" * 68)
    print("  ALGORITHM-FAMILY AVERAGES  (F1 / SHD / time)")
    print("=" * 68)
    num_cols = [c for c in ["f1", "shd", "elapsed_s"] if c in df.columns]
    agg = (
        df.groupby("algo_family")[num_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    print(agg.to_string())

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    X, true_adj, node_names = load_asia(n_samples=5000, seed=42)

    # ── Run all algorithm families ────────────────────────────────────────────
    #
    # Each runner returns a list of result dicts.
    # Algorithms are grouped by how long they typically take:
    #   Fast  (<10 s each)  : PC, GES, ICALiNGAM, DirectLiNGAM
    #   Medium (<5 min)     : ANMNonlinear, Notears, GOLEM
    #   Slow   (5-30 min)   : NotearsNonlinear, GraNDAG, MCSL, GAE, RL, CORL
    #
    # Set the flags below to False to skip algorithm families during development.
    # ──────────────────────────────────────────────────────────────────────────

    RUN_CONFIG = {
        "pc":                True,
        "ges":               True,
        "icalingam":         True,
        "directlingam":      True,
        "anm":               True,
        "notears":           True,
        "notears_nonlinear": True,
        "golem":             True,
        "grandag":           True,
        "mcsl":              True,
        "gae":               True,
        "rl":                True,
        "corl":              True,
    }

    all_records = []

    if RUN_CONFIG["pc"]:
        all_records += run_pc(X, true_adj)
    if RUN_CONFIG["ges"]:
        all_records += run_ges(X, true_adj)
    if RUN_CONFIG["icalingam"]:
        all_records += run_icalingam(X, true_adj)
    if RUN_CONFIG["directlingam"]:
        all_records += run_directlingam(X, true_adj)
    if RUN_CONFIG["anm"]:
        all_records += run_anm(X, true_adj)
    if RUN_CONFIG["notears"]:
        all_records += run_notears(X, true_adj)
    if RUN_CONFIG["notears_nonlinear"]:
        all_records += run_notears_nonlinear(X, true_adj)
    if RUN_CONFIG["golem"]:
        all_records += run_golem(X, true_adj)
    if RUN_CONFIG["grandag"]:
        all_records += run_grandag(X, true_adj)
    if RUN_CONFIG["mcsl"]:
        all_records += run_mcsl(X, true_adj)
    if RUN_CONFIG["gae"]:
        all_records += run_gae(X, true_adj)
    if RUN_CONFIG["rl"]:
        all_records += run_rl(X, true_adj)
    if RUN_CONFIG["corl"]:
        all_records += run_corl(X, true_adj)

    # ── Summarise ─────────────────────────────────────────────────────────────
    results_df = print_summary(all_records)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = "gcastle_asia_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved ➜  {out_path}")

    return results_df


if __name__ == "__main__":
    results = main()
