#!/bin/bash

# Initialize conda
__conda_setup="$($HOME/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Activate conda environment
conda activate cupid

# Function to execute when Ctrl+C is pressed
cleanup() {
    echo "Stopping all background processes..."
    kill $(jobs -p)
    wait
    echo "All processes stopped."
}

# Use trap to catch SIGINT (Ctrl+C)
trap cleanup SIGINT

# Run multiple commands in the background
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/system/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/system/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/system/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/user/GPT-4o-mini-2024-07-18 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/user/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/user/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/user/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/relevant/user/GPT-4o-mini-2024-07-18 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/relevant/user/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/relevant/user/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset zero/relevant/user/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/system/3/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/system/3/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/system/3/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/1/GPT-4o-mini-2024-07-18 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/1/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/1/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/1/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/3/GPT-4o-mini-2024-07-18 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/3/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/3/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/3/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/5/GPT-4o-mini-2024-07-18 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/5/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/5/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/user/5/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/relevant/user/3/GPT-4o-mini-2024-07-18 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/relevant/user/3/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/relevant/user/3/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset few/relevant/user/3/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset bm25/dialogue/1/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset bm25/dialogue/3/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset bm25/dialogue/5/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset bm25/utterance/5/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset bm25/utterance/15/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset bm25/utterance/25/Llama-3.1-8B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset bm25/utterance/5/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset bm25/utterance/5/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset contriever/dialogue/1/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset contriever/dialogue/3/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset contriever/dialogue/5/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset contriever/utterance/5/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset contriever/utterance/15/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset contriever/utterance/25/Llama-3.1-8B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset contriever/utterance/5/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset contriever/utterance/5/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/lr/1e-6/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/lr/3e-6/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/lr/1e-5/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/lr/3e-5/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/lr/1e-4/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/lr/3e-4/Llama-3.1-8B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/rank/8/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/rank/16/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/rank/32/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/rank/64/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/rank/128/Llama-3.1-8B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/lr/1e-4/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset sft/lr/1e-4/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/lr/1e-6/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/lr/3e-6/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/lr/1e-5/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/lr/3e-5/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/lr/1e-4/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/lr/3e-4/Llama-3.1-8B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/rank/8/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/rank/16/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/rank/32/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/rank/64/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/rank/128/Llama-3.1-8B-Instruct &

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/lr/1e-6/Mistral-7B-Instruct-v0.3 &
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu.yaml -m src.evaluate_hf --config configs/evaluate_hf.yaml --dataset dpo/lr/1e-6/Qwen2.5-7B-Instruct &

# Wait for all background processes to finish
wait
