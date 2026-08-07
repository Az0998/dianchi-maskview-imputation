"""Dianchi spatiotemporal imputation benchmark configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "data"
PARQUET = DATA_RAW / "dianchi_data_df.parquet"
DISTANCE_CSV = DATA_RAW / "dianchi_station_distance_km.csv"

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
TABLES = RESULTS / "tables"
CKPT = ROOT / "checkpoints"

VARS = ["TEM", "PH", "DO", "CON", "NTU", "IMN", "NH_N", "TP", "TN"]
FREQ = "4h"

# Sliding windows
SEQ_LEN = 36          # 6 days @ 4h
STRIDE = 18
TRAIN_END = "2023-06-30"
VAL_END = "2023-12-31"

# Graph (spatial_plus_l0 defaults)
K_NEIGHBORS = 8
MAX_DIST_KM = 60.0
GAUSS_SIGMA_KM = 20.0

# Artificial missing rates for evaluation (on originally observed cells)
EVAL_MISSING_RATES = [0.1, 0.2, 0.3]
EVAL_PATTERNS = ["point", "block_time", "sensor", "station", "mixed"]

# Mask-View training
N_VIEWS = 3
MASK_VIEW_PATTERNS = ["point", "block_time", "sensor", "station"]
VIEW_MASK_RATIO = (0.15, 0.35)  # sample ratio range per view

# Training (demo-friendly defaults; raise for paper runs)
SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda"  # auto-fallback in code
HIDDEN = 64
N_HEADS = 4
N_LAYERS = 3  # MixHop ST depth for stronger spatial propagation
LAMBDA_CONS = 0.0  # ablation: higher λ hurts average MAE at current scale
LAMBDA_REC = 1.0

# PyPOTS baselines: keep epochs modest for first end-to-end pass
PYPOTS_EPOCHS = 15
PYPOTS_PATIENCE = 5

# Downstream
DO_DROP_THR = 2.0       # mg/L drop within 24h → anomaly
TN_EXCEED = 2.0         # mg/L Grade-III style proxy
