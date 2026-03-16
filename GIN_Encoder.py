
import os, math, random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as RD, Crippen, Descriptors as Desc
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

# ---------------- Config ----------------
CSV_LABELED = "DE Data Collection.csv"  
CKPT_PATH   = "GIN_checkpoint/pi1m_ssl.ckpt"                               
SMILES_COL  = "RDKit_SMILES"
TARGET_COLS = ["Dielectric Constant", "Young's Modulus (MPa)"]

USE_DESCRIPTORS = True      
EMB_BATCH_SIZE  = 32
SEED = 42
HIDDEN, LAYERS, DROPOUT = 256, 6, 0.1

MASK_ATOM_ID, PAD_ATOM_ID, FIRST_ATOM_ID, MAX_ATOMIC_NUM = 0, 1, 2, 100
from rdkit.Chem import rdchem
BOND_TO_ID = {
    rdchem.BondType.SINGLE: 1,
    rdchem.BondType.DOUBLE: 2,
    rdchem.BondType.TRIPLE: 3,
    rdchem.BondType.AROMATIC: 4,
}

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def atom_to_id(a: Chem.Atom) -> int:
    z = int(a.GetAtomicNum())
    if z <= 0: return MASK_ATOM_ID
    return FIRST_ATOM_ID + min(z, MAX_ATOMIC_NUM)

def bond_to_id(b: Chem.Bond) -> int:
    return BOND_TO_ID.get(b.GetBondType(), 0)

def smiles_to_graph(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0: return None
    x = torch.tensor([[atom_to_id(a)] for a in m.GetAtoms()], dtype=torch.long)
    src, dst, eattr = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = bond_to_id(b)
        src += [i, j]; dst += [j, i]; eattr += [[bt], [bt]]
    if len(src) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr  = torch.empty((0,1), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(eattr, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ---------------- Optional: dielectric-friendly descriptors ----------------
ESTER = Chem.MolFromSmarts("C(=O)O")
CARB  = Chem.MolFromSmarts("C=O")
VINYL = Chem.MolFromSmarts("C=C")
HALO  = Chem.MolFromSmarts("[F,Cl,Br,I]")

def dielectric_descriptors(m):
    n_heavy = RD.CalcNumHeavyAtoms(m)
    denom = max(1, n_heavy)
    n_hetero = RD.CalcNumHeteroatoms(m) / denom
    n_arom   = sum(int(a.GetIsAromatic()) for a in m.GetAtoms()) / denom
    n_conj   = sum(int(b.GetIsConjugated()) for b in m.GetBonds()) / denom
    n_O = sum(1 for a in m.GetAtoms() if a.GetSymbol()=="O") / denom
    n_N = sum(1 for a in m.GetAtoms() if a.GetSymbol()=="N") / denom
    n_halo = len(m.GetSubstructMatches(HALO)) / denom
    n_ester= len(m.GetSubstructMatches(ESTER)) / denom
    n_carb = len(m.GetSubstructMatches(CARB)) / denom
    n_vinyl= len(m.GetSubstructMatches(VINYL)) / denom
    mr  = Crippen.MolMR(m)                 # polarizability proxy
    logp= Crippen.MolLogP(m)
    tpsa= RD.CalcTPSA(m)
    mw  = Desc.MolWt(m)
    mr_norm   = mr / denom
    tpsa_norm = tpsa / denom
    return np.array([
        n_heavy, n_hetero, n_arom, n_conj, n_O, n_N, n_halo,
        n_ester, n_carb, n_vinyl,
        mr, mr_norm, logp, tpsa, tpsa_norm, mw
    ], dtype=float)

class AtomEncoder(nn.Module):
    def __init__(self, hidden, num_atom_tokens=FIRST_ATOM_ID+MAX_ATOMIC_NUM+1):
        super().__init__()
        self.emb = nn.Embedding(num_atom_tokens, hidden)
    def forward(self, x_long): return self.emb(x_long.view(-1))

class GINBackbone(nn.Module):
    def __init__(self, hidden, layers=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.layers.append(GINConv(mlp))
        self.drop = nn.Dropout(dropout)
    def forward(self, x, edge_index, batch):
        for conv in self.layers:
            x = conv(x, edge_index); x = F.relu(x); x = self.drop(x)
        g = global_mean_pool(x, batch)
        return x, g

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AtomEncoder(HIDDEN)
        self.gnn = GINBackbone(HIDDEN, LAYERS, DROPOUT)
    def forward(self, data):  # returns [B,H]
        x0 = self.enc(data.x); _, g = self.gnn(x0, data.edge_index, data.batch); return g

# ---------------- Load data & make features ----------------
seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(CSV_LABELED).dropna(subset=[SMILES_COL]+TARGET_COLS).copy()
mols = [Chem.MolFromSmiles(s) for s in df[SMILES_COL]]
mask = [m is not None for m in mols]
df = df.loc[mask].reset_index(drop=True)
mols = [m for m in mols if m is not None]
Y = df[TARGET_COLS].astype(float).values
N = len(mols)
print(f"Loaded {N} labeled samples.")

# encode with pretrained GNN
enc = Encoder().to(device)
try:
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
except TypeError:
    ckpt = torch.load(CKPT_PATH, map_location=device)  # fallback for older torch
missing, unexpected = enc.load_state_dict(ckpt["model_state"], strict=False)
print("Loaded pretrain (missing, unexpected):", len(missing), len(unexpected))

graphs = []
for m in mols:
    s = Chem.MolToSmiles(m)  # canonicalize
    g = smiles_to_graph(s)
    if g is not None: graphs.append(g)

loader = DataLoader(graphs, batch_size=EMB_BATCH_SIZE, shuffle=False)
X_emb = []
enc.eval()
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        g = enc(batch)                # [B, H]
        X_emb.append(g.cpu().numpy())
X_emb = np.vstack(X_emb)             # [N, H]

if USE_DESCRIPTORS:
    X_desc = np.vstack([dielectric_descriptors(m) for m in mols])  # [N, D]
    X = np.hstack([X_emb, X_desc])                                 # [N, H+D]
else:
    X = X_emb

print("Feature shape:", X.shape, "Targets shape:", Y.shape)

# ---------------- Nested LOOCV with standardised PCA grid ----------------
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PCA_GRID = [5, 10, 15, 20, 25]        # ← standardised across ALL models
BOOTSTRAP_ITERS = 5000

kernel = (
    C(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1))
)

loo = LeaveOneOut()
y_true_all, y_pred_all = [], []
pca_var_ratios = []
selected_pca_components = []

print(f"\nRunning nested LOOCV ({N} folds, inner 5-fold CV)...")
for fold_i, (tr_idx, te_idx) in enumerate(loo.split(X)):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=SEED)),
        ("multi", MultiOutputRegressor(
            GaussianProcessRegressor(
                kernel=kernel, normalize_y=True,
                random_state=SEED, n_restarts_optimizer=3
            )
        ))
    ])

    param_grid = {
        "pca__n_components": PCA_GRID,
        "multi__estimator__alpha": [1e-10, 1e-6, 1e-3],
        "multi__estimator__n_restarts_optimizer": [2, 5],
    }

    cv5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(pipe, param_grid, cv=cv5, scoring="r2", n_jobs=-1, verbose=0)
    grid.fit(X[tr_idx], Y[tr_idx])

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X[te_idx])

    pca_var_ratios.append(best_model.named_steps['pca'].explained_variance_ratio_.sum())
    selected_pca_components.append(best_model.named_steps['pca'].n_components)

    y_true_all.append(Y[te_idx][0])
    y_pred_all.append(y_pred[0])

    if (fold_i + 1) % 10 == 0 or fold_i == 0:
        print(f"  Fold {fold_i+1}/{N}: PCA={best_model.named_steps['pca'].n_components}")

y_true_all = np.vstack(y_true_all)
y_pred_all = np.vstack(y_pred_all)

# PCA summary
print(f"\nPCA component selection across {N} LOOCV folds:")
for nc in sorted(set(selected_pca_components)):
    count = selected_pca_components.count(nc)
    print(f"  n_components={nc}: selected {count}/{N} times")
print(f"PCA explained variance: {np.mean(pca_var_ratios)*100:.1f}% ± {np.std(pca_var_ratios)*100:.1f}%")

# Metrics
mse_per  = mean_squared_error(y_true_all, y_pred_all, multioutput="raw_values")
rmse_per = np.sqrt(mse_per)
r2_per   = r2_score(y_true_all, y_pred_all, multioutput="raw_values")
rmse_mean = float(np.mean(rmse_per))
r2_mean   = float(np.mean(r2_per))

print(f"\nPer-target Metrics (LOO, {N} samples):")
print(f"  k:  RMSE={rmse_per[0]:.4f} | R²={r2_per[0]:.4f}")
print(f"  E:  RMSE={rmse_per[1]:.4f} | R²={r2_per[1]:.4f}")
print(f"Mean: RMSE_mean={rmse_mean:.4f} | R²_mean={r2_mean:.4f}")

# Uncertainty
r2_k_j, r2_E_j, r2_mean_j = [], [], []
for i in range(N):
    mask = np.ones(N, dtype=bool); mask[i] = False
    r2k = r2_score(y_true_all[mask, 0], y_pred_all[mask, 0])
    r2e = r2_score(y_true_all[mask, 1], y_pred_all[mask, 1])
    r2_k_j.append(r2k); r2_E_j.append(r2e); r2_mean_j.append((r2k + r2e) / 2.0)

r2_k_j, r2_E_j, r2_mean_j = np.array(r2_k_j), np.array(r2_E_j), np.array(r2_mean_j)

rng = np.random.default_rng(SEED)
rmse_k_b = np.zeros(BOOTSTRAP_ITERS); rmse_E_b = np.zeros(BOOTSTRAP_ITERS); rmse_mean_b = np.zeros(BOOTSTRAP_ITERS)
for b in range(BOOTSTRAP_ITERS):
    idx = rng.integers(0, N, size=N)
    rk = math.sqrt(mean_squared_error(y_true_all[idx, 0], y_pred_all[idx, 0]))
    re = math.sqrt(mean_squared_error(y_true_all[idx, 1], y_pred_all[idx, 1]))
    rmse_k_b[b] = rk; rmse_E_b[b] = re; rmse_mean_b[b] = (rk + re) / 2.0

print("\n==============================")
print("R²: jackknife mean ± std | RMSE: bootstrap mean ± std\n")
print(f"R² (k):    {r2_k_j.mean():.4f} ± {r2_k_j.std(ddof=1):.4f}")
print(f"R² (E):    {r2_E_j.mean():.4f} ± {r2_E_j.std(ddof=1):.4f}")
print(f"R² (Mean): {r2_mean_j.mean():.4f} ± {r2_mean_j.std(ddof=1):.4f}")
print(f"\nRMSE (k):    {rmse_k_b.mean():.4f} ± {rmse_k_b.std(ddof=1):.4f}")
print(f"RMSE (E):    {rmse_E_b.mean():.4f} ± {rmse_E_b.std(ddof=1):.4f}")
print(f"RMSE (Mean): {rmse_mean_b.mean():.4f} ± {rmse_mean_b.std(ddof=1):.4f}")

