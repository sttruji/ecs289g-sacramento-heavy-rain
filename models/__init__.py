# XGBoost is imported before the PyTorch models on purpose: both bring an OpenMP
# runtime, and on macOS fitting XGBoost after PyTorch has loaded its own libomp
# crashes the process (segmentation fault). XGBoost first, then torch, works.
from models.xgboost_model import XGBoost
from models.climatology import Climatology
from models.gat import GAT
from models.gru import GRU
from models.hrrr import HRRR
from models.logistic_regression import LogisticRegression
from models.persistence import Persistence
from models.random_forest import RandomForest
from models.vae import VAE
from models.seed_average import SeedAverage
