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
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "meta-llama/Llama-3.1-8B-Instruct" --prompt "zero_v1" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "mistralai/Mistral-7B-Instruct-v0.3" --prompt "zero_v1" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "Qwen/Qwen2.5-7B-Instruct" --prompt "zero_v1" &

CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "meta-llama/Llama-3.1-8B-Instruct" --prompt "zero_v2" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "mistralai/Mistral-7B-Instruct-v0.3" --prompt "zero_v2" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "Qwen/Qwen2.5-7B-Instruct" --prompt "zero_v2" &

CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "meta-llama/Llama-3.1-8B-Instruct" --prompt "few_v1" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "mistralai/Mistral-7B-Instruct-v0.3" --prompt "few_v1" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "Qwen/Qwen2.5-7B-Instruct" --prompt "few_v1" &

CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "meta-llama/Llama-3.1-8B-Instruct" --prompt "few_v2" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "mistralai/Mistral-7B-Instruct-v0.3" --prompt "few_v2" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "Qwen/Qwen2.5-7B-Instruct" --prompt "few_v2" &

CUDA_VISIBLE_DEVICES=0 python inference_hf.py --peft_path "peft/sft/lora/Llama-3.1-8B-Instruct" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --peft_path "peft/sft/lora/Mistral-7B-Instruct-v0.3" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --peft_path "peft/sft/lora/Qwen2.5-7B-Instruct" &

CUDA_VISIBLE_DEVICES=0 python inference_hf.py --peft_path "peft/dpo/lora/Llama-3.1-8B-Instruct" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --peft_path "peft/dpo/lora/Mistral-7B-Instruct-v0.3" &
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --peft_path "peft/dpo/lora/Qwen2.5-7B-Instruct" &

# Wait for all background processes to finish
wait
