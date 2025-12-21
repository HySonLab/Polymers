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



CSV_PATH = "DE Data Collection.csv"
SMI_COLS = ["RDKit_SMILES", "SMILES", "SMILE"]
TARGET_COLS = ["Dielectric Constant", "Young's Modulus (MPa)"]


EMBED_PATH = "artifacts/transPolymer_embeddings.pkl"

RANDOM_SEED = 42
BATCH_SIZE = 64 
PCA_GRID = [16, 32, 64, 128, 256, 512]


BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 42



def load_embeddings(path: str) -> np.ndarray:
    """
    Load embeddings from either:
      - .npy -> np.load
      - .pkl/.pickle -> pickle.load

    Supports common pickle containers:
      - np.ndarray
      - dict with key 'embeddings'
      - list of vectors (will vstack)
      - list of dicts with 'embedding' key

    Returns: X float ndarray shape (N, D)
    """
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
            # Check if it's a list of dicts with 'embedding' key
            if obj and isinstance(obj[0], dict) and "embedding" in obj[0]:
                X = np.vstack([item["embedding"] for item in obj])
            else:
                # list of vectors
                X = np.vstack([np.asarray(v) for v in obj])
        else:
            raise ValueError(
                f"Unsupported pickle format in {path}. "
                f"Got type={type(obj)}. Expected ndarray, dict['embeddings'], or list."
            )
    else:
        raise ValueError(f"Unsupported embeddings file extension: {ext} (path={path})")

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (N,D). Got shape={X.shape}")

    return X


def main():
    # --------------------------
    # 1) Load CSV + basic checks
    # --------------------------
    df = pd.read_csv(CSV_PATH)

    smi_col = next((c for c in SMI_COLS if c in df.columns), None)
    if smi_col is None:
        raise ValueError(f"No SMILES column found among {SMI_COLS}. Got: {list(df.columns)}")

    for t in TARGET_COLS:
        if t not in df.columns:
            raise ValueError(f"Missing target column: {t}")

    # Keep only rows with targets and SMILES present
    df = df.dropna(subset=[smi_col] + TARGET_COLS).reset_index(drop=True)

    # Targets
    y = df[TARGET_COLS].to_numpy(dtype=float)

    # --------------------------
    # 2) Load embeddings
    # --------------------------
    X_tp = load_embeddings(EMBED_PATH)

    # Alignment check: embeddings must match filtered CSV row count
    if X_tp.shape[0] != y.shape[0]:
        raise ValueError(
            "Row mismatch between CSV (after dropna) and embeddings.\n"
            f"CSV rows (filtered): {y.shape[0]}\n"
            f"Embeddings rows:      {X_tp.shape[0]}\n\n"
            "Fix: ensure embeddings were generated in the SAME row order as the CSV, "
            "and using the SAME filtering rules."
        )

    print(f"[INFO] Using SMILES column: {smi_col}")
    print(f"[INFO] X shape: {X_tp.shape} | y shape: {y.shape}")

    # --------------------------
    # 3) GPR pipeline + GridSearch (5CV)
    # --------------------------
    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    )

    base_gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=RANDOM_SEED,
        n_restarts_optimizer=5,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(max(16, X_tp.shape[1] // 2), X_tp.shape[1] - 1), random_state=RANDOM_SEED)),
            ("multi", MultiOutputRegressor(base_gpr)),
        ]
    )

    pca_candidates = [n for n in PCA_GRID if n < X_tp.shape[1]] or [min(X_tp.shape[1] - 1, 64)]
    param_grid = {
        "pca__n_components": pca_candidates,
        "multi__estimator__alpha": [1e-10, 1e-6, 1e-3],
        "multi__estimator__n_restarts_optimizer": [2, 5, 10],
    }

    cv5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )

    print("[INFO] GridSearchCV fitting …")
    grid.fit(X_tp, y)
    best = grid.best_estimator_

    print("\nBest params:", grid.best_params_)
    print("Best CV R^2:", grid.best_score_)

    # --------------------------
    # 4) LOO-CV with tuned model
    # --------------------------
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for tr, te in loo.split(X_tp):
        best.fit(X_tp[tr], y[tr])
        y_hat = best.predict(X_tp[te])
        y_true.append(y[te][0])
        y_pred.append(y_hat[0])

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\nOverall Metrics (LOO-CV) on {len(X_tp)} samples:")
    print(f"  MSE  = {mse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R2   = {r2:.4f}")

    # --------------------------
    # 5) Per-target LOO metrics
    # --------------------------
    mse_per = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    mae_per = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    rmse_per = np.sqrt(mse_per)
    r2_per = r2_score(y_true, y_pred, multioutput="raw_values")

    rmse_mean = float(np.mean(rmse_per))
    r2_mean = float(np.mean(r2_per))

    print("\nPer-target Metrics (LOO):")
    print(f"  k:  RMSE={rmse_per[0]:.4f} | R2={r2_per[0]:.4f}")
    print(f"  E:  RMSE={rmse_per[1]:.4f} | R2={r2_per[1]:.4f}")
    print("Mean across targets:")
    print(f"  RMSE_mean={rmse_mean:.4f} | R2_mean={r2_mean:.4f}")

    # --------------------------
    #    - R²: jackknife mean ± std
    #    - RMSE: bootstrap mean ± std
    # --------------------------
    N = len(y_true)

    # Jackknife R² (N leave-one-out subsets of LOOCV predictions)
    r2_k_j, r2_E_j, r2_mean_j = [], [], []
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        r2k = r2_score(y_true[mask, 0], y_pred[mask, 0])
        r2e = r2_score(y_true[mask, 1], y_pred[mask, 1])
        r2_k_j.append(r2k)
        r2_E_j.append(r2e)
        r2_mean_j.append((r2k + r2e) / 2.0)

    r2_k_j = np.asarray(r2_k_j)
    r2_E_j = np.asarray(r2_E_j)
    r2_mean_j = np.asarray(r2_mean_j)

    r2_k_mean_j, r2_k_std_j = r2_k_j.mean(), r2_k_j.std(ddof=1)
    r2_E_mean_j, r2_E_std_j = r2_E_j.mean(), r2_E_j.std(ddof=1)
    r2_mean_mean_j, r2_mean_std_j = r2_mean_j.mean(), r2_mean_j.std(ddof=1)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx_all = np.arange(N)

    rmse_k_b = np.zeros(BOOTSTRAP_B, dtype=float)
    rmse_E_b = np.zeros(BOOTSTRAP_B, dtype=float)
    rmse_mean_b = np.zeros(BOOTSTRAP_B, dtype=float)

    for b in range(BOOTSTRAP_B):
        idx = rng.choice(idx_all, size=N, replace=True)
        rmse_k_tmp = np.sqrt(mean_squared_error(y_true[idx, 0], y_pred[idx, 0]))
        rmse_E_tmp = np.sqrt(mean_squared_error(y_true[idx, 1], y_pred[idx, 1]))
        rmse_k_b[b] = rmse_k_tmp
        rmse_E_b[b] = rmse_E_tmp
        rmse_mean_b[b] = (rmse_k_tmp + rmse_E_tmp) / 2.0

    rmse_k_mean_b, rmse_k_std_b = rmse_k_b.mean(), rmse_k_b.std(ddof=1)
    rmse_E_mean_b, rmse_E_std_b = rmse_E_b.mean(), rmse_E_b.std(ddof=1)
    rmse_mean_mean_b, rmse_mean_std_b = rmse_mean_b.mean(), rmse_mean_b.std(ddof=1)

    print("\n==============================")

    print("R²: jackknife mean ± std from LOOCV predictions.")
    print("RMSE: bootstrap mean ± std by resampling LOOCV (y_true, y_pred) pairs.\n")

    print(f"R² (k):    {r2_k_mean_j:.4f} ± {r2_k_std_j:.4f}")
    print(f"R² (E):    {r2_E_mean_j:.4f} ± {r2_E_std_j:.4f}")
    print(f"R² (Mean): {r2_mean_mean_j:.4f} ± {r2_mean_std_j:.4f}")

    print(f"\nRMSE (k):    {rmse_k_mean_b:.4f} ± {rmse_k_std_b:.4f}")
    print(f"RMSE (E):    {rmse_E_mean_b:.4f} ± {rmse_E_std_b:.4f}")
    print(f"RMSE (Mean): {rmse_mean_mean_b:.4f} ± {rmse_mean_std_b:.4f}")


if __name__ == "__main__":
    main()
