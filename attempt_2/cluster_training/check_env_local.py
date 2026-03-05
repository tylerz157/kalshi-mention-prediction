#!/usr/bin/env python3
"""
check_env_local.py -- Verify library versions locally before sending to cluster.

Run from the cluster_training/ folder:
  python check_env_local.py
"""

import sys
import os
from pathlib import Path

REQUIRED = {
    "torch":          "2.1.0",
    "transformers":   "4.40.0",
    "datasets":       "2.18.0",
    "trl":            "0.8.0",
    "accelerate":     "0.28.0",
}

print(f"Python {sys.version}")
print()

all_ok = True

# -- Library versions --
print("Library versions:")
try:
    from packaging.version import Version
    has_packaging = True
except ImportError:
    has_packaging = False

for pkg, min_ver in REQUIRED.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        if has_packaging:
            ok = Version(ver) >= Version(min_ver)
            status = "OK" if ok else f"WARN (need >= {min_ver})"
            if not ok:
                all_ok = False
        else:
            status = "(install 'packaging' to check version)"
        print(f"  {pkg:<15} {ver:<12} {status}")
    except ImportError:
        print(f"  {pkg:<15} MISSING  (pip install {pkg})")
        all_ok = False

# deepspeed optional locally
print()
try:
    import deepspeed
    print(f"deepspeed:       {deepspeed.__version__}  OK")
except ImportError:
    print("deepspeed:       not installed (only needed on cluster)")

try:
    import flash_attn
    print(f"flash-attn:      {flash_attn.__version__}  OK")
except ImportError:
    print("flash-attn:      not installed (only needed on cluster)")

# -- GPU check (informational only locally) --
print()
print("GPU (informational):")
import torch
if not torch.cuda.is_available():
    print("  No CUDA GPUs -- OK for local check, cluster needs GPUs")
else:
    n = torch.cuda.device_count()
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1024**3
        print(f"  [{i}] {props.name} - {gb:.1f} GB")
    print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")

# -- HF token --
print()
token = os.environ.get("HF_TOKEN", "")
if token.startswith("hf_") and len(token) > 20:
    print(f"HF_TOKEN: set ({token[:8]}...)")
else:
    print("HF_TOKEN: not set (needed on cluster, optional for local check)")

# -- Data files -- check both local and cluster-relative paths --
print()
print("Data files:")
candidates = {
    "openai_train.jsonl": [
        Path("data/openai_train.jsonl"),
        Path("../train/data/openai_train.jsonl"),
    ],
    "openai_val.jsonl": [
        Path("data/openai_val.jsonl"),
        Path("../train/data/openai_val.jsonl"),
    ],
}
for name, paths in candidates.items():
    found = next((p for p in paths if p.exists()), None)
    if found:
        mb = found.stat().st_size / 1024**2
        lines = sum(1 for _ in open(found, encoding="utf-8"))
        print(f"  {name}: {mb:.0f} MB, {lines:,} samples  ({found})  OK")
    else:
        print(f"  {name}: MISSING (copy to cluster_training/data/ before sending)")

# -- Tokenizer smoke test --
print()
print("Tokenizer smoke test (downloads ~500 KB):")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        token=os.environ.get("HF_TOKEN"),
    )
    test = tok.apply_chat_template(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        tokenize=False, add_generation_prompt=False,
    )
    header = "<|start_header_id|>assistant<|end_header_id|>"
    if header in test:
        print("  Chat template OK, response header found")
    else:
        print("  WARN: expected response header not found in template output")
        all_ok = False
except Exception as e:
    print(f"  ERROR: {e}")
    all_ok = False

# -- Summary --
print()
print("=" * 40)
if all_ok:
    print("All checks passed. Safe to send to cluster.")
else:
    print("Some checks FAILED. Review issues above.")
print("=" * 40)
