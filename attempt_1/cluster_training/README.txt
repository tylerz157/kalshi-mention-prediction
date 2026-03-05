SETUP
=====

1. Install dependencies:
   pip install transformers datasets trl accelerate deepspeed flash-attn

2. Get a HuggingFace token with Llama 3.1 access:
   https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
   (request access, then go to https://huggingface.co/settings/tokens)

3. Set your HF token:
   export HF_TOKEN=hf_...

4. Place the data files in a folder called "data/" next to train_llama.py:
   data/openai_train.jsonl   (94 MB)
   data/openai_val.jsonl     (11 MB)

   Directory should look like:
   cluster_training/
   ├── train_llama.py
   ├── deepspeed_config.json
   ├── README.txt
   └── data/
       ├── openai_train.jsonl
       └── openai_val.jsonl

5. Launch training (8 GPUs):
   torchrun --nproc_per_node=8 train_llama.py

   For fewer GPUs (e.g. 4), edit PER_GPU_BATCH or GRAD_ACCUM in train_llama.py
   to keep effective batch size = 64:
     2 GPUs: PER_GPU_BATCH=2, GRAD_ACCUM=16
     4 GPUs: PER_GPU_BATCH=2, GRAD_ACCUM=8

OUTPUT
======
Checkpoints saved to: checkpoints/llama-trump/
Final model saved to: checkpoints/llama-trump/final/

Training: ~10 epochs, ~10,717 samples, batch size 64
Estimated time: ~8-10 hours on 8x A100 80GB

Send back: the entire checkpoints/llama-trump/final/ folder (~16 GB)
