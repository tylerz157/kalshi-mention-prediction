#!/usr/bin/env python3
"""
download_model.py -- Download the fine-tuned model from HuggingFace Hub.

Usage:
  python download_model.py <hf_repo_id>

Example:
  python download_model.py friend-username/llama-trump-finetuned
"""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download

SAVE_DIR = Path("model")

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_model.py <hf_repo_id>")
        print("Example: python download_model.py friend-username/llama-trump-finetuned")
        sys.exit(1)

    repo_id = sys.argv[1]
    token = None
    if len(sys.argv) > 2:
        token = sys.argv[2]

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {SAVE_DIR}/")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(SAVE_DIR),
        token=token,
    )

    print(f"Done. Model saved to: {path}")

if __name__ == "__main__":
    main()
