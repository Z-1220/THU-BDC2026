# Workspace Instructions

This repository is a ranking-based stock selection baseline for the HS300 universe. The core task is to train a `StockTransformer` model that scores and ranks stocks for a given trading day, then outputs the top 5 stocks with equal weights.

## Use when

- working on data preprocessing, feature engineering, or ranking model training
- updating `code/src/train.py`, `code/src/predict.py`, or `code/src/utils.py`
- adding new stock features, loss terms, or ranking metrics
- explaining how to run the project or generate prediction output

## Key files and directories

- `code/src/train.py` — main training script
- `code/src/predict.py` — inference script that outputs `output/result.csv`
- `code/src/config.py` — global config values, file paths, and model settings
- `code/src/model.py` — `StockTransformer` model architecture and related neural network code
- `code/src/utils.py` — feature engineering, dataset construction, and sequence preparation utilities
- `data/train.csv` — default training/prediction input data
- `output/` — model outputs and inference results
- `model/` — saved models and training artifacts
- `Dockerfile` — container environment with TA-Lib and Python dependencies
- `pyproject.toml` — Python dependency specification, including `uv` and PyTorch index settings

## Running the project

Recommended scripts:

- `sh train.sh` → runs `python code/src/train.py`
- `sh test.sh` → runs `python code/src/predict.py`

Environment setup:

- `uv sync` to install dependencies from `pyproject.toml`
- activate `.venv` on Linux/macOS if using the local virtual environment
- `Dockerfile` installs the system TA-Lib dependency and builds a reproducible runtime

## Important project conventions

- The training pipeline uses ranking-based labels derived from future open prices (`open_t1`, `open_t5`) and computes a 5-day return label.
- Feature sets are controlled by `config['feature_num']` and support `'39'` or `'158+39'`.
- `train.py` and `predict.py` both use multiprocessing with `spawn` mode.
- `predict.py` expects `best_model.pth` and `scaler.pkl` in `output/` and writes top-5 results to `./output/result.csv`.
- The model output format is:
  - `stock_id`
  - `weight` (fixed to `0.2` for each selected stock)

## Notes for editing and troubleshooting

- `TA-Lib` is required by feature engineering and may need a system-level install outside Python on Windows or Linux.
- On Windows, PyTorch is configured via the `pytorch-cu128` index in `pyproject.toml`.
- If changing feature engineering or preprocessing, update `code/src/utils.py`, `code/src/train.py`, and `code/src/predict.py` consistently.
- The repository supports Docker-based environment creation, which is useful when native TA-Lib installation fails.

## Helpful prompts

- "Explain the end-to-end training and inference flow in this repository."
- "Help me add a new ranking metric to `code/src/train.py`."
- "Update `README.md` with Docker usage and the recommended training commands."
- "Find and fix any inconsistency between `config.py` and `code/src/predict.py` feature handling."
