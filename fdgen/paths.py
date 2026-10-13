"""Central place for all project paths.

Everything is resolved relative to the repository root, so scripts and
notebooks work no matter which directory they are started from.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- data -------------------------------------------------------------------
DATA_DIR = ROOT / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"                      # labelled FD dataset
COLUMN_DESCRIPTIONS_JSON = DATA_DIR / "column_descriptions.json"
ARCHIVES_DIR = DATA_DIR / "archives"                        # data.zip, results_fd.zip (Git LFS)
TABLES_DIR = DATA_DIR / "tables"                            # source tables (unzipped from data.zip)
FDS_DIR = DATA_DIR / "fds"                                  # HyFD/HyUCC outputs (unzipped from results_fd.zip)
CAR_SALES_DIR = DATA_DIR / "car_sales"
FINETUNING_DIR = DATA_DIR / "finetuning"                    # train/val splits for LLM finetuning

# --- models -----------------------------------------------------------------
MODELS_DIR = ROOT / "models"

# --- outputs ----------------------------------------------------------------
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

# --- misc -------------------------------------------------------------------
SQL_DIR = ROOT / "sql"
TOOLS_DIR = ROOT / "tools"                                  # Metanome / HyFD jars
ENV_FILE = ROOT / ".env"
