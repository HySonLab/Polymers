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
# 1) Load + basic cleaning
# =========================
csv_path = "DE Data Collection.csv"
df = pd.read_csv(csv_path)

# Parse SMILES
df["Mol"] = df["RDKit_SMILES"].apply(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)

# Keep only valid molecules + non-missing targets
target_cols = ["Dielectric Constant", "Young's Modulus (MPa)"]
df = df[df["Mol"].notnull()].copy()
df = df.dropna(subset=target_cols).copy()

print(f"Usable rows after cleaning: {len(df)}")


# ==========================================
# 2) Build Morgan fingerprints (radius=2)
# ==========================================
n_bits = 1024
fps = []
for mol in df["Mol"]:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    fps.append(arr)

X = np.asarray(fps, dtype=float)  # float helps StandardScaler/PCA
y = df[target_cols].values.astype(float)  # two continuous targets


# ==========================================
# 3) GPR pipeline + small grid search (5CV)
# ==========================================
kernel = (
    C(1.0, (1e-3, 1e3))
    * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
)

base_gpr = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    random_state=42,
    n_restarts_optimizer=5
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=50, random_state=42)),
    ("multi", MultiOutputRegressor(base_gpr))
])

param_grid = {
    "pca__n_components": [20, 50, 100],
    "multi__estimator__alpha": [1e-10, 1e-6, 1e-3],
    "multi__estimator__n_restarts_optimizer": [2, 5, 10],
}

cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

grid5 = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv5,
    scoring="r2",  # mean across outputs by default
    n_jobs=-1,
    verbose=1
)

grid5.fit(X, y)
best_model = grid5.best_estimator_

print("\nBest params:", grid5.best_params_)
print("Best 5CV R^2 (mean across targets):", grid5.best_score_)


# ==========================================
# 4) Leave-One-Out evaluation (per-target)
# ==========================================
loo = LeaveOneOut()
y_true_all, y_pred_all = [], []

for train_idx, test_idx in loo.split(X):
    X_tr, X_ts = X[train_idx], X[test_idx]
    y_tr, y_ts = y[train_idx], y[test_idx]

    best_model.fit(X_tr, y_tr)
    y_pred = best_model.predict(X_ts)

    y_true_all.append(y_ts[0])
    y_pred_all.append(y_pred[0])

y_true_all = np.vstack(y_true_all)
y_pred_all = np.vstack(y_pred_all)

# ----------------------
# Overall (aggregated)
# ----------------------
mse_overall  = mean_squared_error(y_true_all, y_pred_all)  # uniform average across targets
mae_overall  = mean_absolute_error(y_true_all, y_pred_all)
rmse_overall = np.sqrt(mse_overall)
r2_overall   = r2_score(y_true_all, y_pred_all)            # uniform average across targets

print(f"\nOverall Metrics (LOO on {len(X)} samples):")
print(f"  MSE  = {mse_overall:.4f}")
print(f"  MAE  = {mae_overall:.4f}")
print(f"  RMSE = {rmse_overall:.4f}")
print(f"  R2   = {r2_overall:.4f}")

# ----------------------
# Per-target (k and E)
# ----------------------
target_names = ["Dielectric Constant (k)", "Young's Modulus (E, MPa)"]

mse_per  = mean_squared_error(y_true_all, y_pred_all, multioutput="raw_values")
mae_per  = mean_absolute_error(y_true_all, y_pred_all, multioutput="raw_values")
rmse_per = np.sqrt(mse_per)
r2_per   = r2_score(y_true_all, y_pred_all, multioutput="raw_values")

mse_mean  = float(np.mean(mse_per))
mae_mean  = float(np.mean(mae_per))
rmse_mean = float(np.mean(rmse_per))
r2_mean   = float(np.mean(r2_per))

print("\nPer-target Metrics (LOO):")
for i, name in enumerate(target_names):
    print(f"  {name}:")
    print(f"    MSE  = {mse_per[i]:.4f}")
    print(f"    MAE  = {mae_per[i]:.4f}")
    print(f"    RMSE = {rmse_per[i]:.4f}")
    print(f"    R2   = {r2_per[i]:.4f}")

print("\nMean across targets (matches Table-mean idea):")
print(f"  MSE_mean  = {mse_mean:.4f}")
print(f"  MAE_mean  = {mae_mean:.4f}")
print(f"  RMSE_mean = {rmse_mean:.4f}")
print(f"  R2_mean   = {r2_mean:.4f}")


# ==========================================================
# 5) Table-ready uncertainty
#    - R²: jackknife (LOO-based) mean ± std  (already good)
#    - RMSE: bootstrap over LOOCV prediction pairs for mean ± std (added)
# ==========================================================
N = len(y_true_all)

# ---- Jackknife R²: compute R² on N-1 points, repeated N times
r2_k_j = []
r2_E_j = []
r2_mean_j = []

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

# ---- Bootstrap RMSE: resample (y_true, y_pred) pairs with replacement
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 42
rng = np.random.default_rng(BOOTSTRAP_SEED)

rmse_k_b = np.zeros(BOOTSTRAP_B, dtype=float)
rmse_E_b = np.zeros(BOOTSTRAP_B, dtype=float)
rmse_mean_b = np.zeros(BOOTSTRAP_B, dtype=float)

idx_all = np.arange(N)
for b in range(BOOTSTRAP_B):
    idx = rng.choice(idx_all, size=N, replace=True)
    rmse_k = np.sqrt(mean_squared_error(y_true_all[idx, 0], y_pred_all[idx, 0]))
    rmse_E = np.sqrt(mean_squared_error(y_true_all[idx, 1], y_pred_all[idx, 1]))
    rmse_k_b[b] = rmse_k
    rmse_E_b[b] = rmse_E
    rmse_mean_b[b] = (rmse_k + rmse_E) / 2.0

rmse_k_mean_b, rmse_k_std_b = rmse_k_b.mean(), rmse_k_b.std(ddof=1)
rmse_E_mean_b, rmse_E_std_b = rmse_E_b.mean(), rmse_E_b.std(ddof=1)
rmse_mean_mean_b, rmse_mean_std_b = rmse_mean_b.mean(), rmse_mean_b.std(ddof=1)

print("\n==============================")
print("Table-ready uncertainty (LOO-based)")
print("==============================")
print("R²: jackknife mean ± std from LOOCV predictions.")
print("RMSE: bootstrap mean ± std by resampling LOOCV (y_true, y_pred) pairs.\n")

print(f"R² (k):    {r2_k_mean_j:.4f} ± {r2_k_std_j:.4f}")
print(f"R² (E):    {r2_E_mean_j:.4f} ± {r2_E_std_j:.4f}")
print(f"R² (Mean): {r2_mean_mean_j:.4f} ± {r2_mean_std_j:.4f}")

print(f"\nRMSE (k):    {rmse_k_mean_b:.4f} ± {rmse_k_std_b:.4f}")
print(f"RMSE (E):    {rmse_E_mean_b:.4f} ± {rmse_E_std_b:.4f}")
print(f"RMSE (Mean): {rmse_mean_mean_b:.4f} ± {rmse_mean_std_b:.4f}")

