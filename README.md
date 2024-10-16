# HiCUPID

## Usage

> [!NOTE]
> Make sure that you are in the project directory.

### 1. Set up virtual environment

Install Miniconda from [here](https://docs.anaconda.com/miniconda/).

```bash
./env_setup.sh
```

### 2. Configure `.env`

```bash
vim .env
```

> [!IMPORTANT]
> Please enter the following API keys in the `.env` file:
>
> OPENAI_API_KEY=\
> HF_TOKEN=

```bash
chmod 600 .env
```

### 3. Run scripts

1. Inference using HuggingFace models

```bash
CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "meta-llama/Llama-3.1-8B-Instruct" --prompt "zero_v1"
```

> [!NOTE]
> You can find other examples in `scripts/inference_hf.sh`

2. Inference using OpenAI models

```bash
python inference_openai.py --model_id "gpt-4o-mini-2024-07-18" --prompt "zero_v1"
```

> [!NOTE]
> You can find other examples in `scripts/inference_openai.sh`

3. Evaluation using HuggingFace models

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_ab_hf.py --load_path "inference/baseline/zero_v1/GPT-4o-mini-2024-07-18"
```

> [!NOTE]
> You can find other examples in `scripts/evaluate_hf.sh`

4. Evaluation using OpenAI models

```bash
python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/baseline/zero_v2/GPT-4o-mini-2024-07-18"
```

> [!NOTE]
> You can find other examples in `scripts/evaluate_openai.sh`

5. Train SFT using HuggingFace `TRL` & `Accelerate`

```bash
accelerate launch --config_file "configs/fsdp_8.yaml" train_sft.py --config "configs/sft/example.yaml"
```

6. Train DPO using HuggingFace `TRL` & `Accelerate`

```bash
accelerate launch --config_file "configs/fsdp_8.yaml" train_dpo.py --config "configs/dpo/example.yaml"
```

7. Train Proxy Evaluation Model using HuggingFace `TRL` & `Accelerate`

```bash
accelerate launch --config_file "configs/fsdp_8.yaml" train_eval.py --config "configs/eval/example.yaml"
```

### 4. Debugging

- When performing inference or evaluation with HuggingFace models, please note the following. You can modify `inference_hf.py` and `evaluate_ab_hf.py`.
    - Depending on the computing environment, the `bfloat16` data type may not be supported. Remove the `torch_dtype=torch.bfloat16` option or change it to a different data type.
    - Depending on the computing environment, `flash_attention_2` may not work. Remove `attn_implementation="flash_attention_2"`.
- When training SFT/DPO/Eval, you can modify the YAML configuration files located in the `configs` folder.
    - If GPU memory is insufficient, adjust `per_device_train_batch_size`.
    - Depending on the computing environment, `flash_attention_2` may not work. Set `attn_implementation` to `null`.
    - Depending on the computing environment, the `liger_kernel` may not work. Set `use_liger` to `false`.
