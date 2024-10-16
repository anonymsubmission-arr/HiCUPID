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
python inference_openai.py --model_id "gpt-4o-mini-2024-07-18" --prompt "zero_v1" &
python inference_openai.py --model_id "gpt-4o-mini-2024-07-18" --prompt "zero_v2" &

python inference_openai.py --model_id "gpt-4o-mini-2024-07-18" --prompt "few_v1" &
python inference_openai.py --model_id "gpt-4o-mini-2024-07-18" --prompt "few_v2" &

# Wait for all background processes to finish
wait
