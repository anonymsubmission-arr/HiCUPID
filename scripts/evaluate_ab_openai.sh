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
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/baseline/zero_v2/GPT-4o-mini-2024-07-18" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/baseline/zero_v2/Llama-3.1-8B-Instruct" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/baseline/zero_v2/Mistral-7B-Instruct-v0.3" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/baseline/zero_v2/Qwen2.5-7B-Instruct" &

python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/baseline/few_v2/GPT-4o-mini-2024-07-18" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/baseline/few_v2/Llama-3.1-8B-Instruct" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/baseline/few_v2/Mistral-7B-Instruct-v0.3" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/baseline/few_v2/Qwen2.5-7B-Instruct" &

python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/bm25/utterance/15/Llama-3.1-8B-Instruct" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/bm25/utterance/15/Mistral-7B-Instruct-v0.3" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/bm25/utterance/15/Qwen2.5-7B-Instruct" &

python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/contriever/utterance/15/Llama-3.1-8B-Instruct" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/contriever/utterance/15/Mistral-7B-Instruct-v0.3" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/contriever/utterance/15/Qwen2.5-7B-Instruct" &

python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/sft/lora/Llama-3.1-8B-Instruct" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/sft/lora/Mistral-7B-Instruct-v0.3" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/sft/lora/Qwen2.5-7B-Instruct" &

python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/dpo/lora/Llama-3.1-8B-Instruct" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/dpo/lora/Mistral-7B-Instruct-v0.3" &
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/dpo/lora/Qwen2.5-7B-Instruct" &

# Wait for all background processes to finish
wait
