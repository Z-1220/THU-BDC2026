# Repository Guidelines

Contributor guide for the THU-BDC2026 quantitative portfolio baseline. The project fine-tunes a Kronos small model on A-share data to rank stocks, then builds a Top-3 portfolio. Python is managed exclusively with **uv** — never use pip, conda, or poetry.

## Project Structure & Module Organization

- `code/` — source code. `code/src/` holds the pipeline (`train.py`, `commit.py`), `code/models/` model implementations (Kronos, LightGBM, transformers), `code/processors/` and `code/handlers/` data and market handlers, `code/PortfolioBuilder/` portfolio logic.
- `scripts/` — data utilities: download stock data, convert to Qlib binary, fine-tune Kronos.
- `test/` — evaluation scripts (`test.py`, `score_docker.py`, `score_self.py`) and the official blind test set.
- `data/` — raw CSVs (`stock_data.csv`, `test.csv`); `temp/qlib_data/` holds generated Qlib binary cache.
- `model/` — experiment config snapshots (`result_model_*.yaml`, `config_snapshot_*.yaml`).
- `docs/` — research reports and experiment plans.

## Build, Test, and Development Commands

- `uv sync` — install dependencies from `pyproject.toml`/`uv.lock`.
- `sh init.sh` — build the Qlib binary data cache under `temp/qlib_data/`.
- `uv run code/src/train.py` — train the model and update config snapshots.
- `uv run code/src/commit.py` — generate the final `result.csv` from the champion config.
- `uv run scripts/convert_data.py` — encode sector features into Qlib data.
- `bash run.sh` — end-to-end pipeline: init → train → test (mirrors the Docker entrypoint).
- `docker build .` / `docker compose up` — containerized run used by the competition.

## Coding Style & Naming Conventions

- Python 3.12, type hints preferred, 4-space indentation, `snake_case` for files/functions, `PascalCase` for classes and module directories (e.g., `PortfolioBuilder`, `KronosModel`).
- Config snapshots follow `result_model_<experiment>.yaml` / `config_snapshot_<experiment>.yaml` naming.
- Keep model checkpoints, generated outputs, and large binaries out of Git (`.gitignore`).

## Testing Guidelines

- Evaluation runs through `test/` scripts with `uv run`; there is no pytest suite.
- Always verify a change end-to-end with `bash test.sh` and report backtest metrics (Sharpe, win rate, max drawdown) when touching model or data code.
- Never modify `data/test.csv` or other official blind-test inputs.

## Commit & Pull Request Guidelines

- Follow Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `perf:`, `refactor:`, `revert:`, `configs:`.
- Use imperative, specific subjects; attach experiment results to model-related commits.
- PRs must link the related issue, summarize the change, and include evaluation metrics or screenshots for behavior-changing work. Push final artifacts to the configured remotes.

## Agent-Specific Instructions

When an agent works here, consult `CLAUDE.md`/`GUIDE.md` for research context, never train on the blind test interval (2026-04-13 to 2026-04-17), and prefer freezing the Kronos tokenizer over fine-tuning it.
