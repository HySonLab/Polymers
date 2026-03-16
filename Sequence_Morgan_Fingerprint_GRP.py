import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
# Config
# =========================
RANDOM_SEED = 42
PCA_GRID = [5, 10, 15, 20, 25]        # ← standardised across ALL models
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 42

csv_path = "DE Data Collection.csv"
target_cols = ["Dielectric Constant", "Young's Modulus (MPa)"]


# =========================
# 1) Load + basic cleaning
# =========================
df = pd.read_csv(csv_path)
df["Mol"] = df["RDKit_SMILES"].apply(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
df = df[df["Mol"].notnull()].copy()
df = df.dropna(subset=target_cols).copy()
print(f"Usable rows after cleaning: {len(df)}")


# =========================
# 2) Morgan fingerprints
# =========================
n_bits = 1024
fps = []
for mol in df["Mol"]:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    fps.append(arr)

X = np.asarray(fps, dtype=float)
y = df[target_cols].values.astype(float)


# =========================
# 3) Nested LOOCV (GridSearchCV inside each fold)
# =========================
kernel = (
    C(1.0, (1e-3, 1e3))
    * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
)

loo = LeaveOneOut()
y_true_all, y_pred_all = [], []
pca_var_ratios = []
selected_pca_components = []

print(f"\nRunning nested LOOCV ({len(X)} folds, inner 5-fold CV)...")
for fold_i, (train_idx, test_idx) in enumerate(loo.split(X)):
    X_tr, X_ts = X[train_idx], X[test_idx]
    y_tr, y_ts = y[train_idx], y[test_idx]

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
    grid.fit(X_tr, y_tr)

    best = grid.best_estimator_
    y_hat = best.predict(X_ts)

    # Track PCA info
    pca_var_ratios.append(best.named_steps['pca'].explained_variance_ratio_.sum())
    selected_pca_components.append(best.named_steps['pca'].n_components)

    y_true_all.append(y_ts[0])
    y_pred_all.append(y_hat[0])

    if (fold_i + 1) % 10 == 0 or fold_i == 0:
        print(f"  Fold {fold_i+1}/{len(X)}: best PCA={best.named_steps['pca'].n_components}, "
              f"best params={grid.best_params_}")

y_true_all = np.vstack(y_true_all)
y_pred_all = np.vstack(y_pred_all)

# PCA summary
print(f"\nPCA component selection across {len(X)} LOOCV folds:")
for nc in sorted(set(selected_pca_components)):
    count = selected_pca_components.count(nc)
    print(f"  n_components={nc}: selected {count}/{len(X)} times")
print(f"PCA explained variance: {np.mean(pca_var_ratios)*100:.1f}% ± {np.std(pca_var_ratios)*100:.1f}%")


# =========================
# 4) Metrics
# =========================
mse_per  = mean_squared_error(y_true_all, y_pred_all, multioutput="raw_values")
rmse_per = np.sqrt(mse_per)
r2_per   = r2_score(y_true_all, y_pred_all, multioutput="raw_values")
rmse_mean = float(np.mean(rmse_per))
r2_mean   = float(np.mean(r2_per))

print(f"\nPer-target Metrics (LOO, {len(X)} samples):")
print(f"  k:  RMSE={rmse_per[0]:.4f} | R²={r2_per[0]:.4f}")
print(f"  E:  RMSE={rmse_per[1]:.4f} | R²={r2_per[1]:.4f}")
print(f"Mean across targets:")
print(f"  RMSE_mean={rmse_mean:.4f} | R²_mean={r2_mean:.4f}")


# =========================
# 5) Uncertainty (jackknife R², bootstrap RMSE)
# =========================
N = len(y_true_all)

r2_k_j, r2_E_j, r2_mean_j = [], [], []
for i in range(N):
    mask = np.ones(N, dtype=bool); mask[i] = False
    r2k = r2_score(y_true_all[mask, 0], y_pred_all[mask, 0])
    r2e = r2_score(y_true_all[mask, 1], y_pred_all[mask, 1])
    r2_k_j.append(r2k); r2_E_j.append(r2e); r2_mean_j.append((r2k + r2e) / 2.0)

r2_k_j, r2_E_j, r2_mean_j = np.array(r2_k_j), np.array(r2_E_j), np.array(r2_mean_j)

rng = np.random.default_rng(BOOTSTRAP_SEED)
rmse_k_b = np.zeros(BOOTSTRAP_B); rmse_E_b = np.zeros(BOOTSTRAP_B); rmse_mean_b = np.zeros(BOOTSTRAP_B)
for b in range(BOOTSTRAP_B):
    idx = rng.choice(N, size=N, replace=True)
    rk = np.sqrt(mean_squared_error(y_true_all[idx, 0], y_pred_all[idx, 0]))
    re = np.sqrt(mean_squared_error(y_true_all[idx, 1], y_pred_all[idx, 1]))
    rmse_k_b[b] = rk; rmse_E_b[b] = re; rmse_mean_b[b] = (rk + re) / 2.0

print("\n==============================")
print("R²: jackknife mean ± std | RMSE: bootstrap mean ± std\n")
print(f"R² (k):    {r2_k_j.mean():.4f} ± {r2_k_j.std(ddof=1):.4f}")
print(f"R² (E):    {r2_E_j.mean():.4f} ± {r2_E_j.std(ddof=1):.4f}")
print(f"R² (Mean): {r2_mean_j.mean():.4f} ± {r2_mean_j.std(ddof=1):.4f}")
print(f"\nRMSE (k):    {rmse_k_b.mean():.4f} ± {rmse_k_b.std(ddof=1):.4f}")
print(f"RMSE (E):    {rmse_E_b.mean():.4f} ± {rmse_E_b.std(ddof=1):.4f}")
print(f"RMSE (Mean): {rmse_mean_b.mean():.4f} ± {rmse_mean_b.std(ddof=1):.4f}")
