#!/usr/bin/env python3
"""
convert_to_gguf.py -- Convert the safetensors model to GGUF Q4 format for fast CPU inference.

Usage:
  python convert_to_gguf.py

Requires:
  pip install llama-cpp-python transformers torch
"""

import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path("model")
GGUF_PATH = Path("model/model-q4.gguf")


def main():
    if GGUF_PATH.exists():
        print(f"GGUF already exists: {GGUF_PATH}")
        print("Delete it first if you want to reconvert.")
        return

    # Use the llama.cpp convert script bundled with transformers
    print("Converting model to GGUF format...")
    print("This may take a few minutes...\n")

    try:
        # Method 1: Use transformers' built-in convert
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print("[1/3] Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float16,
            device_map="cpu",
        )

        print("[2/3] Saving as fp16 for conversion...")
        fp16_dir = MODEL_DIR / "fp16_temp"
        fp16_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(fp16_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(fp16_dir))

        print("[3/3] Converting to GGUF Q4_K_M...")
        # Try using llama-cpp-python's convert script
        result = subprocess.run(
            [sys.executable, "-m", "llama_cpp.llama_convert",
             "--outfile", str(GGUF_PATH),
             "--outtype", "q4_k_m",
             str(fp16_dir)],
            capture_output=True, text=True,
        )

        if result.returncode != 0:
            # Fallback: try convert_hf_to_gguf from llama.cpp
            print("Built-in convert not available, trying alternative...")
            result = subprocess.run(
                [sys.executable, "-c",
                 f"from llama_cpp import Llama; "
                 f"Llama.from_pretrained('{MODEL_DIR}', "
                 f"filename='model-q4.gguf')"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print("Auto-conversion failed.")
                print("\nManual alternative:")
                print("  1. Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
                print("  2. pip install -r llama.cpp/requirements.txt")
                print(f"  3. python llama.cpp/convert_hf_to_gguf.py {MODEL_DIR} --outfile {GGUF_PATH} --outtype q4_k_m")
                return

        # Cleanup temp files
        import shutil
        shutil.rmtree(str(fp16_dir), ignore_errors=True)

        print(f"\nDone! GGUF saved to: {GGUF_PATH}")
        size_gb = GGUF_PATH.stat().st_size / 1024**3
        print(f"Size: {size_gb:.1f} GB (was 15 GB)")

    except Exception as e:
        print(f"Error: {e}")
        print("\nManual alternative:")
        print("  1. Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
        print("  2. pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
        print(f"  3. python llama.cpp/convert_hf_to_gguf.py {MODEL_DIR} --outfile {GGUF_PATH} --outtype q4_k_m")


if __name__ == "__main__":
    main()
