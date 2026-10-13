# Genuine Functional Dependencies

Data profiling algorithms such as HyFD find *every* functional dependency (FD) that holds on a table. Most of them hold by accident.
This project builds a labelled dataset of discovered FDs and trains and evaluates classifiers (feature-based models and LLMs) that tell **genuine** FDs apart from **accidental** ones.

## Repository layout

```
.
├── main.py                  # CLI pipeline: transform → split → classify → evaluate
├── fdgen/                   # Python package with all reusable code
│   ├── paths.py             #   every project path, resolved from the repo root
│   ├── labelling.py         #   helpers for reading FD results and writing labels
│   ├── data_transformer.py  #   feature / description transformers
│   ├── classifiers.py       #   random + LLM classifiers
│   ├── llm_utils.py         #   Ollama / Perplexity calls and prompts
│   └── evaluate_fd.py       #   precision / recall / F1 evaluation
├── notebooks/
│   ├── labelling/           # per-dataset FD inspection and labelling, SQL generation
│   ├── analysis/            # features, visualisation, pruning policy, EDA
│   └── llm/                 # column descriptions, finetuning, Ollama tests
├── data/
│   ├── dataset.csv          # ★ the labelled FD dataset
│   ├── column_descriptions.json
│   ├── archives/            # data.zip, results_fd.zip (Git LFS)
│   ├── tables/              # source tables, incl. car_sales + mondialjoin (unzipped, git-ignored)
│   ├── fds/                 # HyFD / HyUCC results  (unzipped, git-ignored)
│   ├── car_sales/           # car sales case study: discovered + relevant FDs
│   └── finetuning/          # train / val splits for LLM finetuning
├── models/                  # tokenizer + tokenised finetuning datasets
├── outputs/                 # figures, evaluation runs, pruning_output.csv
├── sql/                     # SQL used to build the joined source tables
└── tools/                   # Metanome / HyFD / HyUCC jars (git-ignored)
```

Code never uses paths relative to the working directory. Import them from `fdgen.paths` instead:

```python
from fdgen.paths import DATASET_CSV, TABLES_DIR, FDS_DIR
df = pd.read_csv(DATASET_CSV)
```

## Setup

```bash
# 1. Python package + dependencies (editable, so notebooks can `import fdgen`)
pip install -e .              # add [finetune] for torch / transformers / datasets / peft

# 2. Large files via Git LFS
git lfs install
git lfs pull

# 3. Unpack source tables and FD results into place (only needed for notebooks that read them)
unzip -j data/archives/data.zip       -d data/tables
unzip -j data/archives/results_fd.zip -d data/fds
```

LLM settings (`LLM_MODEL`, `OLLAMA_API_URL`, `TIMEOUT`, `TEMPERATURE`, `PERPLEXITY_API_KEY`) are read from `.env` in the repo root.

## Running the pipeline

```bash
python main.py --data data/dataset.csv --classifier random
python main.py --data data/dataset.csv --metadata data/column_descriptions.json --classifier llm
```

Splits and predictions are written to `outputs/evaluation/` (override with `--outputdir`).

## Dataset: `data/dataset.csv`

568 labelled FDs from five tables: northwind, croppedairline, financial, women_basketball and mondialjoin.
118 are genuine and 450 are not.

| Column | Meaning |
|---|---|
| `id` | Running row id |
| `lhs` | Left-hand-side attributes, comma-separated (e.g. `TaxiOut, DayofMonth`) |
| `rhs` | Right-hand-side attribute |
| `tablename` | Source table the FD was discovered on |
| `is_approx` | `True` if the FD came from approximate discovery |
| `null_eq_nul` | Whether NULL = NULL semantics were used during discovery |
| `is_genuine` | Label: `True` = genuine, `False` = accidental |

New labels are appended with `fdgen.labelling.writeBackTrainingData`. It refuses to add a table that is already in the file.

## Ollama

Local LLM inference uses [Ollama](https://ollama.com):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

## Troubleshooting

**`End-of-central-directory signature not found` when unzipping** or **a ZIP of only a few KB:** the file is still a Git LFS pointer. Run `git lfs pull`.

**`ModuleNotFoundError: No module named 'fdgen'` in a notebook:** run `pip install -e .` in the environment the notebook kernel uses.

**`HTTP 400` / `RPC failed` on push:** run `git config http.postBuffer 157286400`.
