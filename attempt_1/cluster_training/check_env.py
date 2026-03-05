#!/usr/bin/env python3
"""
check_env.py -- Verify library versions and GPU setup before training.

Run this BEFORE train_llama.py to catch compatibility issues early.
  python check_env.py
"""

import sys

REQUIRED = {
    "torch":          "2.1.0",
    "transformers":   "4.40.0",
    "datasets":       "2.18.0",
    "trl":            "0.8.0",
    "accelerate":     "0.28.0",
    "deepspeed":      "0.14.0",
}

print(f"Python {sys.version}")
print()

all_ok = True

# -- Library versions --
print("Library versions:")
for pkg, min_ver in REQUIRED.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        from packaging.version import Version
        ok = Version(ver) >= Version(min_ver)
        status = "OK" if ok else f"WARN (need >= {min_ver})"
        if not ok:
            all_ok = False
        print(f"  {pkg:<15} {ver:<12} {status}")
    except ImportError:
        print(f"  {pkg:<15} MISSING")
        all_ok = False

# flash-attn is optional but recommended
print()
try:
    import flash_attn
    print(f"flash-attn:      {flash_attn.__version__}  (optional, OK)")
except ImportError:
    print("flash-attn:      not installed (optional, training will still work)")

# -- GPU check --
print()
print("GPU:")
import torch
if not torch.cuda.is_available():
    print("  ERROR: No CUDA GPUs detected")
    all_ok = False
else:
    n = torch.cuda.device_count()
    print(f"  {n} GPU(s) detected")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1024**3
        print(f"  [{i}] {props.name} - {gb:.1f} GB")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")

# -- DeepSpeed quick check --
print()
print("DeepSpeed:")
try:
    import deepspeed
    print(f"  version {deepspeed.__version__} installed")
    print("  Run 'ds_report' in terminal for full diagnostic")
except Exception as e:
    print(f"  ERROR: {e}")
    all_ok = False

# -- HF token check --
print()
import os
token = os.environ.get("HF_TOKEN", "")
if token.startswith("hf_") and len(token) > 20:
    print(f"HF_TOKEN: set ({token[:8]}...)")
else:
    print("HF_TOKEN: NOT SET - required for Llama 3.1 (export HF_TOKEN=hf_...)")
    all_ok = False

# -- Data files check --
print()
print("Data files:")
from pathlib import Path
for f in ["data/openai_train.jsonl", "data/openai_val.jsonl"]:
    p = Path(f)
    if p.exists():
        mb = p.stat().st_size / 1024**2
        lines = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  {f}: {mb:.0f} MB, {lines:,} samples  OK")
    else:
        print(f"  {f}: MISSING")
        all_ok = False

# -- Quick tokenizer smoke test (no model download) --
print()
print("Tokenizer smoke test:")
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
        print(f"  Chat template OK, response header found")
    else:
        print(f"  WARN: expected header not found in template output")
        all_ok = False
except Exception as e:
    print(f"  ERROR: {e}")
    all_ok = False

# -- Summary --
print()
print("=" * 40)
if all_ok:
    print("All checks passed. Ready to train.")
    print("Run: torchrun --nproc_per_node=8 train_llama.py")
else:
    print("Some checks FAILED. Fix the issues above before training.")
print("=" * 40)
