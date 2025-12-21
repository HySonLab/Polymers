
import math
import pickle
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


CSV_PATH = "DE Data Collection.csv"
EMBED_PKL = "Polybert_Embeddings.pkl" 
TARGET_COLS = ["Dielectric Constant", "Young's Modulus (MPa)"]
RANDOM_SEED = 42

BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 42


PREDICTION_CSV_OUT = "predictions_gp_polybert.csv"



def load_embeddings_from_pkl(pkl_path: str) -> np.ndarray:
    """
    Loads embeddings from a PKL file.

    Expected format (as in your code):
        embed_data = pickle.load(...)
        X_list = [item['embedding'] for item in embed_data]

    Returns:
        X: np.ndarray of shape (N, D)
    """
    with open(pkl_path, "rb") as f:
        embed_data = pickle.load(f)

    # Assumes list[dict] with "embedding" key (same as your current logic)
    X_list = [item["embedding"] for item in embed_data]

    if not X_list:
        raise ValueError("No embeddings found in PKL file.")

    dims = {len(v) for v in X_list}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions found: {dims}")

    X = np.vstack(X_list).astype(float)
    return X


def main():
    # --------------------------
    # 1) Load CSV targets
    # --------------------------
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Keep only rows with targets available (unchanged intention)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    y = df[TARGET_COLS].to_numpy(dtype=float)

    # --------------------------
    # 2) Load embeddings
    # --------------------------
    X = load_embeddings_from_pkl(EMBED_PKL)

    # Safety check: embeddings must align with CSV row order
    if len(X) != len(y):
        raise ValueError(
            f"Row mismatch: embeddings have {len(X)} rows but CSV targets have {len(y)} rows.\n"
            f"Fix by ensuring the PKL embeddings are saved in EXACTLY the same row order as the CSV."
        )

    print(f"[INFO] Loaded targets: y shape={y.shape}")
    print(f"[INFO] Loaded embeddings: X shape={X.shape}")

    # --------------------------
    # 3) GPR pipeline + GridSearchCV (5-fold)
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

    # PCA components safety cap (same logic as your script)
    n_samples, n_features = X.shape
    max_pca_safe = max(5, min(n_features - 1, int(0.8 * n_samples) - 1))
    candidate_pca = [10, 15, 20, 25, 30]
    pca_grid = sorted({k for k in candidate_pca if k <= max_pca_safe})
    if not pca_grid:
        pca_grid = [min(10, max_pca_safe)]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(20, max_pca_safe), random_state=RANDOM_SEED)),
            ("multi", MultiOutputRegressor(base_gpr)),
        ]
    )

    param_grid = {
        "pca__n_components": pca_grid,
        "multi__estimator__alpha": [1e-10, 1e-6, 1e-3],
        "multi__estimator__n_restarts_optimizer": [2, 5, 10],
    }

    cv5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid5 = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )

    print("[INFO] Running 5-fold GridSearchCV...")
    grid5.fit(X, y)

    best_model = grid5.best_estimator_
    print("\nBest params (5-fold CV):", grid5.best_params_)
    print("Best CV R^2 (mean across outputs):", grid5.best_score_)

    # --------------------------
    # 4) LOOCV evaluation (tuned pipeline)
    # --------------------------
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for tr_idx, te_idx in loo.split(X):
        best_model.fit(X[tr_idx], y[tr_idx])
        y_hat = best_model.predict(X[te_idx])

        y_true_all.append(y[te_idx][0])
        y_pred_all.append(y_hat[0])

    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    # Overall metrics (same definitions as your code)
    mse_overall = mean_squared_error(y_true_all, y_pred_all)
    mae_overall = mean_absolute_error(y_true_all, y_pred_all)
    rmse_overall = math.sqrt(mse_overall)
    r2_overall = r2_score(y_true_all, y_pred_all)

    print(f"\nOverall Metrics (LOO-CV) on {len(X)} samples:")
    print(f"  MSE  = {mse_overall:.4f}")
    print(f"  MAE  = {mae_overall:.4f}")
    print(f"  RMSE = {rmse_overall:.4f}")
    print(f"  R2   = {r2_overall:.4f}")

    # Per-target metrics (same print style)
    for j, tname in enumerate(TARGET_COLS):
        mse_j = mean_squared_error(y_true_all[:, j], y_pred_all[:, j])
        mae_j = mean_absolute_error(y_true_all[:, j], y_pred_all[:, j])
        rmse_j = math.sqrt(mse_j)
        r2_j = r2_score(y_true_all[:, j], y_pred_all[:, j])
        print(f"\nTarget: {tname}")
        print(f"  MSE  = {mse_j:.4f}")
        print(f"  MAE  = {mae_j:.4f}")
        print(f"  RMSE = {rmse_j:.4f}")
        print(f"  R2   = {r2_j:.4f}")

    # Save LOOCV predictions
    out = pd.DataFrame(
        {
            "y_true_dielectric": y_true_all[:, 0],
            "y_pred_dielectric": y_pred_all[:, 0],
            "y_true_youngs_mpa": y_true_all[:, 1],
            "y_pred_youngs_mpa": y_pred_all[:, 1],
        }
    )
    out.to_csv(PREDICTION_CSV_OUT, index=False)
    print(f"\nSaved per-fold predictions to {PREDICTION_CSV_OUT}")

    # --------------------------
    #    - R²: jackknife mean ± std from LOOCV predictions
    #    - RMSE: bootstrap mean ± std by resampling LOOCV pairs
    # --------------------------
    N = len(y_true_all)

    # ---- Jackknife R²
    r2_k_j, r2_E_j, r2_mean_j = [], [], []
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        r2k = r2_score(y_true_all[mask, 0], y_pred_all[mask, 0])
        r2e = r2_score(y_true_all[mask, 1], y_pred_all[mask, 1])
        r2_k_j.append(r2k)
        r2_E_j.append(r2e)
        r2_mean_j.append((r2k + r2e) / 2.0)

    r2_k_j = np.array(r2_k_j)
    r2_E_j = np.array(r2_E_j)
    r2_mean_j = np.array(r2_mean_j)

    r2_k_mean_j, r2_k_std_j = r2_k_j.mean(), r2_k_j.std(ddof=1)
    r2_E_mean_j, r2_E_std_j = r2_E_j.mean(), r2_E_j.std(ddof=1)
    r2_mean_mean_j, r2_mean_std_j = r2_mean_j.mean(), r2_mean_j.std(ddof=1)

    # ---- Bootstrap RMSE on LOOCV pairs
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx_all = np.arange(N)

    rmse_k_b = np.zeros(BOOTSTRAP_B, dtype=float)
    rmse_E_b = np.zeros(BOOTSTRAP_B, dtype=float)
    rmse_mean_b = np.zeros(BOOTSTRAP_B, dtype=float)

    for b in range(BOOTSTRAP_B):
        idx = rng.choice(idx_all, size=N, replace=True)
        rmse_k_tmp = np.sqrt(mean_squared_error(y_true_all[idx, 0], y_pred_all[idx, 0]))
        rmse_E_tmp = np.sqrt(mean_squared_error(y_true_all[idx, 1], y_pred_all[idx, 1]))
        rmse_k_b[b] = rmse_k_tmp
        rmse_E_b[b] = rmse_E_tmp
        rmse_mean_b[b] = (rmse_k_tmp + rmse_E_tmp) / 2.0

    rmse_k_mean_b, rmse_k_std_b = rmse_k_b.mean(), rmse_k_b.std(ddof=1)
    rmse_E_mean_b, rmse_E_std_b = rmse_E_b.mean(), rmse_E_b.std(ddof=1)
    rmse_mean_mean_b, rmse_mean_std_b = rmse_mean_b.mean(), rmse_mean_b.std(ddof=1)


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
