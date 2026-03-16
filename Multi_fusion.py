import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import random
import os
import pickle
warnings.filterwarnings('ignore')

# ============================================================================
# CRITICAL: REPRODUCIBILITY SETUP
# ============================================================================
def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    """Initialize DataLoader workers with different seeds"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============================================================================
# Configuration
# ============================================================================
MASTER_SEED = 42
DATA_CSV = "DE Data Collection.csv"
TP_PATH  = "artifacts/transPolymer_embeddings.pkl"
GNN_PATH = "artifacts/gin_embeddings.pkl"
TARGET_COLUMNS = ["Dielectric Constant", "Young's Modulus (MPa)"]

TEMPERATURE = 0.10
BATCH_SIZE = 16
EPOCHS = 400
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
PROPERTY_PERCENTILE = 30
PROJECTION_DIM = 128
TP_HIDDEN_DIM = 128
GNN_HIDDEN_DIM = 256
TP_DROPOUT = 0.3
GNN_DROPOUT = 0.15

PCA_GRID = [5, 10, 15, 20, 25]   # ← standardised across ALL models
GPR_RESTARTS = 10
NUM_RUNS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Data Loading
# ============================================================================
def load_targets(csv_path: str, target_cols):
    df = pd.read_csv(csv_path)
    cols_lower = [c.lower().strip() for c in df.columns]
    resolved = []
    for tc in target_cols:
        tc_l = tc.lower().strip()
        if tc_l in cols_lower:
            idx = cols_lower.index(tc_l)
            resolved.append(df.columns[idx])
        else:
            raise ValueError(f"Could not find target column '{tc}'")
    y = df[resolved].to_numpy().astype(np.float32)
    return y, resolved

def load_embeddings(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        X = np.load(path)
    elif ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, np.ndarray):
            X = obj
        elif isinstance(obj, dict) and "embeddings" in obj:
            X = np.asarray(obj["embeddings"])
        elif isinstance(obj, list):
            if obj and isinstance(obj[0], dict) and "embedding" in obj[0]:
                X = np.vstack([item["embedding"] for item in obj])
            else:
                X = np.vstack([np.asarray(v) for v in obj])
        else:
            raise ValueError(f"Unsupported pickle format in {path}.")
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Embeddings must be 2D. Got shape={X.shape}")
    return X

def load_data():
    X_tp = load_embeddings(TP_PATH).astype(np.float32)
    X_gnn = load_embeddings(GNN_PATH).astype(np.float32)
    y, resolved_cols = load_targets(DATA_CSV, TARGET_COLUMNS)
    if not (X_tp.shape[0] == X_gnn.shape[0] == y.shape[0]):
        raise ValueError("Row mismatch in data")
    return X_tp, X_gnn, y, resolved_cols

# ============================================================================
# Models
# ============================================================================
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class ContrastiveAlignmentModel(nn.Module):
    def __init__(self, tp_dim, gnn_dim):
        super().__init__()
        self.tp_proj = ProjectionHead(tp_dim, TP_HIDDEN_DIM, PROJECTION_DIM, TP_DROPOUT)
        self.gnn_proj = ProjectionHead(gnn_dim, GNN_HIDDEN_DIM, PROJECTION_DIM, GNN_DROPOUT)
    def forward(self, x_tp, x_gnn):
        return self.tp_proj(x_tp), self.gnn_proj(x_gnn)

def property_guided_loss(z_tp, z_gnn, y):
    y_norm = (y - y.min(dim=0)[0]) / (y.max(dim=0)[0] - y.min(dim=0)[0] + 1e-8)
    prop_dist = torch.cdist(y_norm, y_norm, p=2)
    threshold = torch.quantile(prop_dist.flatten(), PROPERTY_PERCENTILE / 100.0)
    pos_mask = (prop_dist <= threshold).float()
    pos_mask.fill_diagonal_(0)
    pos_counts = pos_mask.sum(dim=1)
    sim = torch.matmul(z_tp, z_gnn.T) / TEMPERATURE
    sim = torch.clamp(sim, -50, 50)
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True)[0])
    pos_sim = (exp_sim * pos_mask).sum(dim=1)
    all_sim = exp_sim.sum(dim=1)
    valid = (pos_counts > 0).float()
    if valid.sum() == 0:
        return (sim * 0.0).sum()
    loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8))
    return (loss * valid).sum() / (valid.sum() + 1e-8)

def train_alignment(X_tp, X_gnn, y, seed, verbose=False):
    set_all_seeds(seed)
    dataset = TensorDataset(
        torch.FloatTensor(X_tp),
        torch.FloatTensor(X_gnn),
        torch.FloatTensor(y)
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed)
    )
    model = ContrastiveAlignmentModel(X_tp.shape[1], X_gnn.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_loss = float('inf')
    patience, counter = 50, 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = []
        for batch in dataloader:
            x_tp, x_gnn, y_batch = [b.to(DEVICE) for b in batch]
            z_tp, z_gnn = model(x_tp, x_gnn)
            loss = property_guided_loss(z_tp, z_gnn, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(loss.item())
        scheduler.step()

        avg_loss = float(np.mean(epoch_loss))
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1

        if verbose and (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}: Loss={avg_loss:.4f}")
        if counter >= patience:
            if verbose:
                print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model

def extract_aligned_embeddings(model, X_tp, X_gnn):
    model.eval()
    with torch.no_grad():
        z_tp, z_gnn = model(
            torch.FloatTensor(X_tp).to(DEVICE),
            torch.FloatTensor(X_gnn).to(DEVICE)
        )
    return z_tp.cpu().numpy(), z_gnn.cpu().numpy()

# ============================================================================
# Fusion Methods
# ============================================================================
def fuse_concatenation(Z_tp, Z_gnn):
    return np.hstack([Z_tp, Z_gnn])

def fuse_averaging(Z_tp, Z_gnn):
    return (Z_tp + Z_gnn) / 2.0

def fuse_weighted(Z_tp, Z_gnn, alpha=0.5):
    return alpha * Z_tp + (1.0 - alpha) * Z_gnn

# ============================================================================
# PCA selection via inner CV (within each LOOCV fold)
# ============================================================================
def select_pca_inner_cv(X_tr, y_tr, seed=42, n_inner_folds=5):
    """Select best PCA n_components via inner k-fold CV on the training set."""
    best_score = -np.inf
    best_k = PCA_GRID[0]
    inner_cv = KFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)

    for k in PCA_GRID:
        if k >= X_tr.shape[0] or k >= X_tr.shape[1]:
            continue
        scores = []
        for inner_tr, inner_val in inner_cv.split(X_tr):
            scaler = StandardScaler()
            X_itr = scaler.fit_transform(X_tr[inner_tr])
            X_ival = scaler.transform(X_tr[inner_val])

            pca = PCA(n_components=k, random_state=seed)
            X_itr_pca = pca.fit_transform(X_itr)
            X_ival_pca = pca.transform(X_ival)

            kernel = (C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
                      + WhiteKernel(1e-3, (1e-6, 1e1)))
            gpr = MultiOutputRegressor(
                GaussianProcessRegressor(
                    kernel=kernel, normalize_y=True, random_state=seed,
                    n_restarts_optimizer=GPR_RESTARTS,
                )
            )
            gpr.fit(X_itr_pca, y_tr[inner_tr])
            y_hat = gpr.predict(X_ival_pca)
            score = r2_score(y_tr[inner_val], y_hat)
            scores.append(score)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    return best_k

# ============================================================================
# Evaluation with inner PCA CV + GPR uncertainty
# ============================================================================
def evaluate_loocv(X, y, name="", seed=42):
    """LOOCV with inner CV for PCA selection + GPR uncertainty."""
    set_all_seeds(seed)
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)
    y_std = np.zeros_like(y)
    pca_selected = []
    pca_var_ratios = []

    for tr_idx, te_idx in loo.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        # Inner CV to select PCA components
        best_k = select_pca_inner_cv(X[tr_idx], y[tr_idx], seed=seed)
        n_components = min(best_k, X_tr.shape[1], X_tr.shape[0] - 1)
        pca_selected.append(n_components)

        pca = PCA(n_components=n_components, random_state=seed)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)
        pca_var_ratios.append(pca.explained_variance_ratio_.sum())

        kernel = (C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)) + WhiteKernel(1e-3, (1e-6, 1e1)))
        gpr = MultiOutputRegressor(
            GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, random_state=seed,
                n_restarts_optimizer=GPR_RESTARTS,
            )
        )
        gpr.fit(X_tr_pca, y[tr_idx])
        y_pred[te_idx] = gpr.predict(X_te_pca)

        # Capture GPR uncertainty (per-target std)
        stds = []
        for est in gpr.estimators_:
            _, s = est.predict(X_te_pca, return_std=True)
            stds.append(s[0])
        y_std[te_idx] = np.array(stds)

    r2_k = r2_score(y[:, 0], y_pred[:, 0])
    r2_E = r2_score(y[:, 1], y_pred[:, 1])
    r2_mean = (r2_k + r2_E) / 2.0
    rmse_k = np.sqrt(mean_squared_error(y[:, 0], y_pred[:, 0]))
    rmse_E = np.sqrt(mean_squared_error(y[:, 1], y_pred[:, 1]))
    rmse_mean = (rmse_k + rmse_E) / 2.0

    return {
        "name": name,
        "r2_k": r2_k, "r2_E": r2_E, "r2_mean": r2_mean,
        "rmse_k": rmse_k, "rmse_E": rmse_E, "rmse_mean": rmse_mean,
        "y_pred": y_pred, "y_std": y_std,
        "pca_selected": pca_selected, "pca_var_ratios": pca_var_ratios,
    }


def evaluate_loocv_early_avg(X_tp, X_gnn, y, name="", seed=42):
    """Early fusion averaging baseline with inner PCA CV."""
    set_all_seeds(seed)
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)
    y_std = np.zeros_like(y)

    for tr_idx, te_idx in loo.split(X_tp):
        sc_tp = StandardScaler()
        Xtp_tr = sc_tp.fit_transform(X_tp[tr_idx])
        Xtp_te = sc_tp.transform(X_tp[te_idx])

        sc_gn = StandardScaler()
        Xgn_tr = sc_gn.fit_transform(X_gnn[tr_idx])
        Xgn_te = sc_gn.transform(X_gnn[te_idx])

        # Use inner CV on concatenated features to pick PCA, then apply to each
        X_concat_tr = np.hstack([X_tp[tr_idx], X_gnn[tr_idx]])
        best_k = select_pca_inner_cv(X_concat_tr, y[tr_idx], seed=seed)
        k = min(best_k, Xtp_tr.shape[1], Xgn_tr.shape[1], Xtp_tr.shape[0] - 1)

        pca_tp = PCA(n_components=k, random_state=seed)
        Ztp_tr = pca_tp.fit_transform(Xtp_tr)
        Ztp_te = pca_tp.transform(Xtp_te)

        pca_gn = PCA(n_components=k, random_state=seed)
        Zgn_tr = pca_gn.fit_transform(Xgn_tr)
        Zgn_te = pca_gn.transform(Xgn_te)

        Z_tr = (Ztp_tr + Zgn_tr) / 2.0
        Z_te = (Ztp_te + Zgn_te) / 2.0

        kernel = (C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)) + WhiteKernel(1e-3, (1e-6, 1e1)))
        gpr = MultiOutputRegressor(
            GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, random_state=seed,
                n_restarts_optimizer=GPR_RESTARTS,
            )
        )
        gpr.fit(Z_tr, y[tr_idx])
        y_pred[te_idx] = gpr.predict(Z_te)

        stds = []
        for est in gpr.estimators_:
            _, s = est.predict(Z_te, return_std=True)
            stds.append(s[0])
        y_std[te_idx] = np.array(stds)

    r2_k = r2_score(y[:, 0], y_pred[:, 0])
    r2_E = r2_score(y[:, 1], y_pred[:, 1])
    r2_mean = (r2_k + r2_E) / 2.0
    rmse_k = np.sqrt(mean_squared_error(y[:, 0], y_pred[:, 0]))
    rmse_E = np.sqrt(mean_squared_error(y[:, 1], y_pred[:, 1]))
    rmse_mean = (rmse_k + rmse_E) / 2.0

    return {
        "name": name,
        "r2_k": r2_k, "r2_E": r2_E, "r2_mean": r2_mean,
        "rmse_k": rmse_k, "rmse_E": rmse_E, "rmse_mean": rmse_mean,
        "y_pred": y_pred, "y_std": y_std,
    }


def evaluate_loocv_true_late_fusion_raw(X_tp, X_gnn, y, alpha=0.5, name="", seed=42):
    """TRUE late fusion: prediction-level combination using raw embeddings."""
    set_all_seeds(seed)
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)
    y_std = np.zeros_like(y)

    for tr_idx, te_idx in loo.split(X_tp):
        # --- TP branch ---
        sc_tp = StandardScaler()
        Xtp_tr = sc_tp.fit_transform(X_tp[tr_idx])
        Xtp_te = sc_tp.transform(X_tp[te_idx])

        k_tp = select_pca_inner_cv(X_tp[tr_idx], y[tr_idx], seed=seed)
        k_tp = min(k_tp, Xtp_tr.shape[1], Xtp_tr.shape[0] - 1)
        pca_tp = PCA(n_components=k_tp, random_state=seed)
        Ztp_tr = pca_tp.fit_transform(Xtp_tr)
        Ztp_te = pca_tp.transform(Xtp_te)

        kernel = (C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)) + WhiteKernel(1e-3, (1e-6, 1e1)))
        gpr_tp = MultiOutputRegressor(
            GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, random_state=seed,
                n_restarts_optimizer=GPR_RESTARTS,
            )
        )
        gpr_tp.fit(Ztp_tr, y[tr_idx])
        y_hat_tp = gpr_tp.predict(Ztp_te)

        # --- GNN branch ---
        sc_gn = StandardScaler()
        Xgn_tr = sc_gn.fit_transform(X_gnn[tr_idx])
        Xgn_te = sc_gn.transform(X_gnn[te_idx])

        k_gn = select_pca_inner_cv(X_gnn[tr_idx], y[tr_idx], seed=seed)
        k_gn = min(k_gn, Xgn_tr.shape[1], Xgn_tr.shape[0] - 1)
        pca_gn = PCA(n_components=k_gn, random_state=seed)
        Zgn_tr = pca_gn.fit_transform(Xgn_tr)
        Zgn_te = pca_gn.transform(Xgn_te)

        gpr_gn = MultiOutputRegressor(
            GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, random_state=seed,
                n_restarts_optimizer=GPR_RESTARTS,
            )
        )
        gpr_gn.fit(Zgn_tr, y[tr_idx])
        y_hat_gn = gpr_gn.predict(Zgn_te)

        y_pred[te_idx] = alpha * y_hat_tp + (1.0 - alpha) * y_hat_gn

        # Combined uncertainty
        stds_tp, stds_gn = [], []
        for est in gpr_tp.estimators_:
            _, s = est.predict(Ztp_te, return_std=True)
            stds_tp.append(s[0])
        for est in gpr_gn.estimators_:
            _, s = est.predict(Zgn_te, return_std=True)
            stds_gn.append(s[0])
        stds_tp = np.array(stds_tp)
        stds_gn = np.array(stds_gn)
        y_std[te_idx] = np.sqrt((alpha * stds_tp)**2 + ((1-alpha) * stds_gn)**2)

    r2_k = r2_score(y[:, 0], y_pred[:, 0])
    r2_E = r2_score(y[:, 1], y_pred[:, 1])
    r2_mean = (r2_k + r2_E) / 2.0
    rmse_k = np.sqrt(mean_squared_error(y[:, 0], y_pred[:, 0]))
    rmse_E = np.sqrt(mean_squared_error(y[:, 1], y_pred[:, 1]))
    rmse_mean = (rmse_k + rmse_E) / 2.0

    return {
        "name": name,
        "r2_k": r2_k, "r2_E": r2_E, "r2_mean": r2_mean,
        "rmse_k": rmse_k, "rmse_E": rmse_E, "rmse_mean": rmse_mean,
        "y_pred": y_pred, "y_std": y_std,
    }

# ============================================================================
# Statistical Analysis
# ============================================================================
def compute_statistics(results_list):
    """Compute mean and std from list of results"""
    metrics = ['r2_k', 'r2_E', 'r2_mean', 'rmse_k', 'rmse_E', 'rmse_mean']
    stats_dict = {}
    for metric in metrics:
        values = [r[metric] for r in results_list]
        stats_dict[f'{metric}_mean'] = np.mean(values)
        stats_dict[f'{metric}_std'] = np.std(values)
    return stats_dict

def perform_statistical_test(results1, results2, metric='r2_mean'):
    """Perform paired t-test AND Wilcoxon test between two methods"""
    values1 = [r[metric] for r in results1]
    values2 = [r[metric] for r in results2]

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(values1, values2)
    diff = np.array(values1) - np.array(values2)
    cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(values1, values2)
    except ValueError:
        # If all differences are zero
        w_stat, p_wilcoxon = 0.0, 1.0

    return {
        't_statistic': t_stat, 'p_ttest': p_ttest,
        'w_statistic': w_stat, 'p_wilcoxon': p_wilcoxon,
        'cohen_d': cohen_d,
        'significant_ttest': p_ttest < 0.05,
        'significant_wilcoxon': p_wilcoxon < 0.05,
    }

# ============================================================================
# Parity Plots with Confidence Intervals
# ============================================================================
def plot_parity_with_ci(y_true, y_pred, y_std, target_names, title, filename):
    """Generate parity plot with 95% confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, (ax, tname) in enumerate(zip(axes, target_names)):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ys = y_std[:, i] * 1.96  # 95% CI

        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))

        ax.errorbar(yt, yp, yerr=ys, fmt='o', color='#2196F3', ecolor='#90CAF9',
                     elinewidth=1.5, capsize=3, markersize=6, alpha=0.8,
                     label=f'Predictions (R²={r2:.3f})')

        lims = [min(yt.min(), yp.min()) - 0.1 * abs(yt.min()),
                max(yt.max(), yp.max()) + 0.1 * abs(yt.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Ideal (y=x)')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Experimental', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'{tname}\nR²={r2:.3f}, RMSE={rmse:.3f}', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# ============================================================================
# Main Experimental Pipeline
# ============================================================================
def main():
    print("="*80)
    print("FULL PIPELINE (v2: nested PCA CV, Wilcoxon test, GPR uncertainty)")
    print("="*80)
    print(f"Master seed: {MASTER_SEED}")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"PCA grid: {PCA_GRID}")
    print(f"Device: {DEVICE}")

    # Load data
    print("\n[1] Loading data...")
    X_tp, X_gnn, y, used_targets = load_data()
    print(f"    TransPolymer: {X_tp.shape}, GNN: {X_gnn.shape}, Targets: {y.shape}")

    print(f"\n[2] Target data statistics:")
    print(f"    {used_targets[0]}: mean={y[:, 0].mean():.3f}, std={y[:, 0].std():.3f}")
    print(f"    {used_targets[1]}: mean={y[:, 1].mean():.3f}, std={y[:, 1].std():.3f}")

    # Storage for all results
    all_results = {
        'early_concat': [],
        'early_avg': [],
        'aligned_concat': [],
        'aligned_avg': [],
        'weighted': {},
        'true_late_raw': {}
    }

    alphas_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    for alpha in alphas_to_test:
        all_results['weighted'][alpha] = []
        all_results['true_late_raw'][alpha] = []

    # Run experiments
    print(f"\n[3] Running {NUM_RUNS} experiments...")
    for run_idx in range(NUM_RUNS):
        seed = MASTER_SEED + run_idx
        print(f"\n  Run {run_idx+1}/{NUM_RUNS} (seed={seed})")
        set_all_seeds(seed)

        # Early fusion baseline: concatenation
        X_early_concat = np.hstack([X_tp, X_gnn])
        res_early = evaluate_loocv(X_early_concat, y, "Early Fusion (Concat)", seed)
        all_results['early_concat'].append(res_early)
        print(f"    Early Fusion (Concat): R²={res_early['r2_mean']:.3f}")

        # Early fusion baseline: averaging
        res_early_avg = evaluate_loocv_early_avg(X_tp, X_gnn, y, "Early Fusion (Avg)", seed)
        all_results['early_avg'].append(res_early_avg)
        print(f"    Early Fusion (Avg):    R²={res_early_avg['r2_mean']:.3f}")

        # TRUE Late Fusion (RAW, prediction-level)
        print(f"    True Late Fusion (RAW) α: ", end="")
        for alpha in alphas_to_test:
            res_late_raw = evaluate_loocv_true_late_fusion_raw(
                X_tp, X_gnn, y, alpha=alpha,
                name=f"True Late RAW α={alpha}", seed=seed
            )
            all_results['true_late_raw'][alpha].append(res_late_raw)
            print(f"{alpha}→{res_late_raw['r2_mean']:.3f} ", end="")
        print()

        # Train alignment
        print(f"    Training alignment...")
        model = train_alignment(X_tp, X_gnn, y, seed, verbose=False)
        Z_tp, Z_gnn = extract_aligned_embeddings(model, X_tp, X_gnn)

        # Aligned concatenation
        X_aligned_concat = fuse_concatenation(Z_tp, Z_gnn)
        res_concat = evaluate_loocv(X_aligned_concat, y, "Aligned Concat", seed)
        all_results['aligned_concat'].append(res_concat)
        print(f"    Aligned Concat: R²={res_concat['r2_mean']:.3f}")

        # Aligned averaging
        X_aligned_avg = fuse_averaging(Z_tp, Z_gnn)
        res_avg = evaluate_loocv(X_aligned_avg, y, "Aligned Avg", seed)
        all_results['aligned_avg'].append(res_avg)
        print(f"    Aligned Avg:   R²={res_avg['r2_mean']:.3f}")

        # Weighted fusion in aligned space
        print(f"    Testing aligned-space α values: ", end="")
        for alpha in alphas_to_test:
            X_weighted = fuse_weighted(Z_tp, Z_gnn, alpha)
            res_weighted = evaluate_loocv(X_weighted, y, f"Aligned Weighted α={alpha}", seed)
            all_results['weighted'][alpha].append(res_weighted)
            print(f"{alpha}→{res_weighted['r2_mean']:.3f} ", end="")
        print()

    # ================================================================
    # Compute statistics
    # ================================================================
    print(f"\n{'='*80}")
    print("RESULTS AGGREGATION")
    print(f"{'='*80}")

    stats_early = compute_statistics(all_results['early_concat'])
    stats_early_avg = compute_statistics(all_results['early_avg'])
    stats_concat = compute_statistics(all_results['aligned_concat'])
    stats_avg = compute_statistics(all_results['aligned_avg'])

    # Best alpha (aligned weighted)
    alpha_stats = {}
    best_alpha, best_alpha_r2 = None, -1
    print("\n[4] Aligned-space alpha value comparison:")
    for alpha in alphas_to_test:
        alpha_stats[alpha] = compute_statistics(all_results['weighted'][alpha])
        mean_r2 = alpha_stats[alpha]['r2_mean_mean']
        std_r2 = alpha_stats[alpha]['r2_mean_std']
        print(f"    α={alpha:.1f}: R² = {mean_r2:.3f} ± {std_r2:.3f}")
        if mean_r2 > best_alpha_r2:
            best_alpha_r2 = mean_r2
            best_alpha = alpha
    print(f"\n    → Best aligned-space α = {best_alpha:.1f} (R² = {best_alpha_r2:.3f})")
    stats_best_weighted = alpha_stats[best_alpha]

    # Best alpha (late fusion)
    late_raw_stats = {}
    best_alpha_raw, best_alpha_raw_r2 = None, -1
    print("\n[5] TRUE late fusion (RAW) alpha value comparison:")
    for alpha in alphas_to_test:
        late_raw_stats[alpha] = compute_statistics(all_results['true_late_raw'][alpha])
        mean_r2 = late_raw_stats[alpha]['r2_mean_mean']
        std_r2 = late_raw_stats[alpha]['r2_mean_std']
        print(f"    α={alpha:.1f}: R² = {mean_r2:.3f} ± {std_r2:.3f}")
        if mean_r2 > best_alpha_raw_r2:
            best_alpha_raw_r2 = mean_r2
            best_alpha_raw = alpha
    print(f"\n    → Best TRUE late-fusion RAW α = {best_alpha_raw:.1f} (R² = {best_alpha_raw_r2:.3f})")
    stats_best_true_late_raw = late_raw_stats[best_alpha_raw]

    # ================================================================
    # PCA explained variance summary (from the last run's best config)
    # ================================================================
    print(f"\n{'='*80}")
    print("[6] PCA EXPLAINED VARIANCE (from last run)")
    print(f"{'='*80}")
    last_aligned_avg = all_results['aligned_avg'][-1]
    if 'pca_var_ratios' in last_aligned_avg and last_aligned_avg['pca_var_ratios']:
        print(f"  Aligned Avg: {np.mean(last_aligned_avg['pca_var_ratios'])*100:.1f}% ± "
              f"{np.std(last_aligned_avg['pca_var_ratios'])*100:.1f}% variance retained")
        print(f"  PCA components selected: {last_aligned_avg.get('pca_selected', 'N/A')}")
    last_early_concat = all_results['early_concat'][-1]
    if 'pca_var_ratios' in last_early_concat and last_early_concat['pca_var_ratios']:
        print(f"  Early Concat: {np.mean(last_early_concat['pca_var_ratios'])*100:.1f}% ± "
              f"{np.std(last_early_concat['pca_var_ratios'])*100:.1f}% variance retained")

    # ================================================================
    # Statistical significance tests
    # ================================================================
    print(f"\n{'='*80}")
    print("[7] STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*80}")

    comparisons = [
        ("Aligned Avg vs Early Concat", all_results['aligned_avg'], all_results['early_concat']),
        ("Aligned Avg vs Early Avg", all_results['aligned_avg'], all_results['early_avg']),
        ("Aligned Concat vs Early Concat", all_results['aligned_concat'], all_results['early_concat']),
        ("Aligned Avg vs Late Fusion (best)", all_results['aligned_avg'],
         all_results['true_late_raw'][best_alpha_raw]),
    ]

    for comp_name, res1, res2 in comparisons:
        test_result = perform_statistical_test(res1, res2, metric='r2_mean')
        sig_t = "✓" if test_result['significant_ttest'] else "✗"
        sig_w = "✓" if test_result['significant_wilcoxon'] else "✗"
        print(f"\n  {comp_name}:")
        print(f"    Paired t-test:     t={test_result['t_statistic']:.3f}, "
              f"p={test_result['p_ttest']:.4f} {sig_t}")
        print(f"    Wilcoxon signed:   W={test_result['w_statistic']:.1f}, "
              f"p={test_result['p_wilcoxon']:.4f} {sig_w}")
        print(f"    Cohen's d:         {test_result['cohen_d']:.3f}")

    # ================================================================
    # Parity Plots with Confidence Intervals
    # ================================================================
    print(f"\n{'='*80}")
    print("[8] PARITY PLOTS WITH CONFIDENCE INTERVALS")
    print(f"{'='*80}")

    # Use the last run's predictions for the best config
    best_result = all_results['aligned_avg'][-1]
    plot_parity_with_ci(
        y, best_result['y_pred'], best_result['y_std'],
        used_targets,
        "Latent-Space Aligned Early Fusion (Averaging)",
        "parity_plot_aligned_avg.png"
    )

    best_early = all_results['early_concat'][-1]
    plot_parity_with_ci(
        y, best_early['y_pred'], best_early['y_std'],
        used_targets,
        "Early Fusion (Concatenation)",
        "parity_plot_early_concat.png"
    )

    # ================================================================
    # Full Results Table
    # ================================================================
    print(f"\n{'='*80}")
    print("Table 3: Fusion Strategy Comparison")
    print(f"{'='*80}\n")

    rows = []

    def pm(mean, std, fmt_mean="{:.3f}", fmt_std="{:.3f}"):
        return f"{fmt_mean.format(mean)} ± {fmt_std.format(std)}"

    def pm_rmse(mean, std):
        return f"{mean:.3f} ± {std:.3f}"

    rows.append({
        "Fusion Type": "Early Fusion", "Method": "Concatenation",
        "R² (k)↑": pm(stats_early['r2_k_mean'], stats_early['r2_k_std']),
        "R² (E)↑": pm(stats_early['r2_E_mean'], stats_early['r2_E_std']),
        "R² (Mean)↑": pm(stats_early['r2_mean_mean'], stats_early['r2_mean_std']),
        "RMSE (k)↓": pm_rmse(stats_early['rmse_k_mean'], stats_early['rmse_k_std']),
        "RMSE (E)↓": pm_rmse(stats_early['rmse_E_mean'], stats_early['rmse_E_std']),
        "RMSE (Mean)↓": pm_rmse(stats_early['rmse_mean_mean'], stats_early['rmse_mean_std']),
    })
    rows.append({
        "Fusion Type": "Early Fusion", "Method": "Averaging",
        "R² (k)↑": pm(stats_early_avg['r2_k_mean'], stats_early_avg['r2_k_std']),
        "R² (E)↑": pm(stats_early_avg['r2_E_mean'], stats_early_avg['r2_E_std']),
        "R² (Mean)↑": pm(stats_early_avg['r2_mean_mean'], stats_early_avg['r2_mean_std']),
        "RMSE (k)↓": pm_rmse(stats_early_avg['rmse_k_mean'], stats_early_avg['rmse_k_std']),
        "RMSE (E)↓": pm_rmse(stats_early_avg['rmse_E_mean'], stats_early_avg['rmse_E_std']),
        "RMSE (Mean)↓": pm_rmse(stats_early_avg['rmse_mean_mean'], stats_early_avg['rmse_mean_std']),
    })
    rows.append({
        "Fusion Type": "Latent-Space Aligned", "Method": "Concatenation",
        "R² (k)↑": pm(stats_concat['r2_k_mean'], stats_concat['r2_k_std']),
        "R² (E)↑": pm(stats_concat['r2_E_mean'], stats_concat['r2_E_std']),
        "R² (Mean)↑": pm(stats_concat['r2_mean_mean'], stats_concat['r2_mean_std']),
        "RMSE (k)↓": pm_rmse(stats_concat['rmse_k_mean'], stats_concat['rmse_k_std']),
        "RMSE (E)↓": pm_rmse(stats_concat['rmse_E_mean'], stats_concat['rmse_E_std']),
        "RMSE (Mean)↓": pm_rmse(stats_concat['rmse_mean_mean'], stats_concat['rmse_mean_std']),
    })
    rows.append({
        "Fusion Type": "Latent-Space Aligned", "Method": "Averaging",
        "R² (k)↑": pm(stats_avg['r2_k_mean'], stats_avg['r2_k_std']),
        "R² (E)↑": pm(stats_avg['r2_E_mean'], stats_avg['r2_E_std']),
        "R² (Mean)↑": pm(stats_avg['r2_mean_mean'], stats_avg['r2_mean_std']),
        "RMSE (k)↓": pm_rmse(stats_avg['rmse_k_mean'], stats_avg['rmse_k_std']),
        "RMSE (E)↓": pm_rmse(stats_avg['rmse_E_mean'], stats_avg['rmse_E_std']),
        "RMSE (Mean)↓": pm_rmse(stats_avg['rmse_mean_mean'], stats_avg['rmse_mean_std']),
    })
    rows.append({
        "Fusion Type": "Late Fusion", "Method": f"Weighted Combination (Aligned, α={best_alpha})",
        "R² (k)↑": pm(stats_best_weighted['r2_k_mean'], stats_best_weighted['r2_k_std']),
        "R² (E)↑": pm(stats_best_weighted['r2_E_mean'], stats_best_weighted['r2_E_std']),
        "R² (Mean)↑": pm(stats_best_weighted['r2_mean_mean'], stats_best_weighted['r2_mean_std']),
        "RMSE (k)↓": pm_rmse(stats_best_weighted['rmse_k_mean'], stats_best_weighted['rmse_k_std']),
        "RMSE (E)↓": pm_rmse(stats_best_weighted['rmse_E_mean'], stats_best_weighted['rmse_E_std']),
        "RMSE (Mean)↓": pm_rmse(stats_best_weighted['rmse_mean_mean'], stats_best_weighted['rmse_mean_std']),
    })
    rows.append({
        "Fusion Type": "Late Fusion (True)", "Method": f"Weighted Prediction (Raw, α={best_alpha_raw})",
        "R² (k)↑": pm(stats_best_true_late_raw['r2_k_mean'], stats_best_true_late_raw['r2_k_std']),
        "R² (E)↑": pm(stats_best_true_late_raw['r2_E_mean'], stats_best_true_late_raw['r2_E_std']),
        "R² (Mean)↑": pm(stats_best_true_late_raw['r2_mean_mean'], stats_best_true_late_raw['r2_mean_std']),
        "RMSE (k)↓": pm_rmse(stats_best_true_late_raw['rmse_k_mean'], stats_best_true_late_raw['rmse_k_std']),
        "RMSE (E)↓": pm_rmse(stats_best_true_late_raw['rmse_E_mean'], stats_best_true_late_raw['rmse_E_std']),
        "RMSE (Mean)↓": pm_rmse(stats_best_true_late_raw['rmse_mean_mean'], stats_best_true_late_raw['rmse_mean_std']),
    })

    df_pub = pd.DataFrame(rows)
    df_pub = df_pub[[
        "Fusion Type", "Method",
        "R² (k)↑", "R² (E)↑", "R² (Mean)↑",
        "RMSE (k)↓", "RMSE (E)↓", "RMSE (Mean)↓"
    ]]
    print(df_pub.to_string(index=False))

if __name__ == "__main__":
    set_all_seeds(MASTER_SEED)
    main()
