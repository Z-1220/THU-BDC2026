"""Download Kronos pretrained models from HuggingFace for offline use.

Downloads: Kronos-Tokenizer-base, Kronos-small, Kronos-base
Saves to: model/kronos_pretrained/

Supports HF_ENDPOINT env var for mirror (e.g. https://hf-mirror.com).
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_DIR = PROJECT_ROOT / "model" / "kronos_pretrained"

MODELS = [
    "NeoQuasar/Kronos-Tokenizer-base",
    "NeoQuasar/Kronos-small",
    "NeoQuasar/Kronos-base",
]


def download_model(repo_id: str, save_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    local_dir = save_dir / repo_id.split("/")[-1]
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[SKIP] {repo_id} already exists at {local_dir}")
        return

    print(f"[DOWNLOAD] {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=[".git", ".gitattributes", "*.md", "LICENSE"],
    )
    print(f"[DONE] {repo_id}")


def main() -> int:
    print("=" * 60)
    print("  Kronos Model Download")
    print(f"  Target: {PRETRAINED_DIR}")
    if os.environ.get("HF_ENDPOINT"):
        print(f"  HF Mirror: {os.environ['HF_ENDPOINT']}")
    print("=" * 60)

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        try:
            download_model(model, PRETRAINED_DIR)
        except Exception as e:
            print(f"[FAIL] {model}: {e}", file=sys.stderr)

    print("\nDone. Models saved to:", PRETRAINED_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
