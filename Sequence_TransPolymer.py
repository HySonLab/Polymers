import os
import math
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
# Config
# =========================
CSV_PATH = "DE Data Collection.csv"
SMI_COLS = ["RDKit_SMILES", "SMILES", "SMILE"]
TARGET_COLS = ["Dielectric Constant", "Young's Modulus (MPa)"]
EMBED_PATH = "artifacts/transPolymer_embeddings.pkl"

RANDOM_SEED = 42
PCA_GRID = [5, 10, 15, 20, 25]        # ← standardised across ALL models
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 42


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


def main():
    # 1) Load CSV + embeddings
    df = pd.read_csv(CSV_PATH)
    smi_col = next((c for c in SMI_COLS if c in df.columns), None)
    if smi_col is None:
        raise ValueError(f"No SMILES column found among {SMI_COLS}.")
    for t in TARGET_COLS:
        if t not in df.columns:
            raise ValueError(f"Missing target column: {t}")
    df = df.dropna(subset=[smi_col] + TARGET_COLS).reset_index(drop=True)
    y = df[TARGET_COLS].to_numpy(dtype=float)

    X_tp = load_embeddings(EMBED_PATH)
    if X_tp.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: embeddings {X_tp.shape[0]} vs CSV {y.shape[0]}")

    print(f"[INFO] X shape: {X_tp.shape} | y shape: {y.shape}")

    # 2) Nested LOOCV
    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    )

    loo = LeaveOneOut()
    y_true, y_pred = [], []
    pca_var_ratios = []
    selected_pca_components = []

    print(f"\nRunning nested LOOCV ({len(X_tp)} folds, inner 5-fold CV)...")
    for fold_i, (tr, te) in enumerate(loo.split(X_tp)):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=RANDOM_SEED)),
            ("multi", MultiOutputRegressor(
                GaussianProcessRegressor(
                    kernel=kernel, normalize_y=True,
                    random_state=RANDOM_SEED, n_restarts_optimizer=5
                )
            ))
        ])

        param_grid = {
            "pca__n_components": PCA_GRID,
            "multi__estimator__alpha": [1e-10, 1e-6, 1e-3],
            "multi__estimator__n_restarts_optimizer": [2, 5, 10],
        }

        cv5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        grid = GridSearchCV(pipe, param_grid, cv=cv5, scoring="r2", n_jobs=-1, verbose=0)
        grid.fit(X_tp[tr], y[tr])

        best = grid.best_estimator_
        y_hat = best.predict(X_tp[te])

        pca_var_ratios.append(best.named_steps['pca'].explained_variance_ratio_.sum())
        selected_pca_components.append(best.named_steps['pca'].n_components)

        y_true.append(y[te][0])
        y_pred.append(y_hat[0])

        if (fold_i + 1) % 10 == 0 or fold_i == 0:
            print(f"  Fold {fold_i+1}/{len(X_tp)}: PCA={best.named_steps['pca'].n_components}")

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # PCA summary
    print(f"\nPCA component selection across {len(X_tp)} LOOCV folds:")
    for nc in sorted(set(selected_pca_components)):
        count = selected_pca_components.count(nc)
        print(f"  n_components={nc}: selected {count}/{len(X_tp)} times")
    print(f"PCA explained variance: {np.mean(pca_var_ratios)*100:.1f}% ± {np.std(pca_var_ratios)*100:.1f}%")

    # 3) Metrics
    mse_per = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    rmse_per = np.sqrt(mse_per)
    r2_per = r2_score(y_true, y_pred, multioutput="raw_values")
    rmse_mean = float(np.mean(rmse_per))
    r2_mean = float(np.mean(r2_per))

    print(f"\nPer-target Metrics (LOO, {len(X_tp)} samples):")
    print(f"  k:  RMSE={rmse_per[0]:.4f} | R²={r2_per[0]:.4f}")
    print(f"  E:  RMSE={rmse_per[1]:.4f} | R²={r2_per[1]:.4f}")
    print(f"Mean: RMSE_mean={rmse_mean:.4f} | R²_mean={r2_mean:.4f}")

    # 4) Uncertainty
    N = len(y_true)
    r2_k_j, r2_E_j, r2_mean_j = [], [], []
    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        r2k = r2_score(y_true[mask, 0], y_pred[mask, 0])
        r2e = r2_score(y_true[mask, 1], y_pred[mask, 1])
        r2_k_j.append(r2k); r2_E_j.append(r2e); r2_mean_j.append((r2k + r2e) / 2.0)

    r2_k_j, r2_E_j, r2_mean_j = np.array(r2_k_j), np.array(r2_E_j), np.array(r2_mean_j)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rmse_k_b = np.zeros(BOOTSTRAP_B); rmse_E_b = np.zeros(BOOTSTRAP_B); rmse_mean_b = np.zeros(BOOTSTRAP_B)
    for b in range(BOOTSTRAP_B):
        idx = rng.choice(N, size=N, replace=True)
        rk = np.sqrt(mean_squared_error(y_true[idx, 0], y_pred[idx, 0]))
        re = np.sqrt(mean_squared_error(y_true[idx, 1], y_pred[idx, 1]))
        rmse_k_b[b] = rk; rmse_E_b[b] = re; rmse_mean_b[b] = (rk + re) / 2.0

    print("\n==============================")
    print("R²: jackknife mean ± std | RMSE: bootstrap mean ± std\n")
    print(f"R² (k):    {r2_k_j.mean():.4f} ± {r2_k_j.std(ddof=1):.4f}")
    print(f"R² (E):    {r2_E_j.mean():.4f} ± {r2_E_j.std(ddof=1):.4f}")
    print(f"R² (Mean): {r2_mean_j.mean():.4f} ± {r2_mean_j.std(ddof=1):.4f}")
    print(f"\nRMSE (k):    {rmse_k_b.mean():.4f} ± {rmse_k_b.std(ddof=1):.4f}")
    print(f"RMSE (E):    {rmse_E_b.mean():.4f} ± {rmse_E_b.std(ddof=1):.4f}")
    print(f"RMSE (Mean): {rmse_mean_b.mean():.4f} ± {rmse_mean_b.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
