# 💖 HiCUPID

We introduce 💖 **HiCUPID**, a new benchmark designed to train and evaluate Large Language Models (LLMs) for **personalized AI assistant** applications. 💖 **HiCUPID** addresses the lack of open-source conversational datasets for personalization by providing a tailored dataset and an automated evaluation model based on [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), which closely aligns with human preferences. Both the **[dataset](https://huggingface.co/datasets/arranonymsub/HiCUPID)** and **[evaluation model](https://huggingface.co/arranonymsub/Llama-3.2-3B-HiCUPID)** are available through a [HuggingFace](https://huggingface.co/arranonymsub) repository, with the code to reproduce the results published on [GitHub](https://github.com/anonymsubmission-arr/HiCUPID). For more details, please refer to our paper *"Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis."*

---

## 📑 Table of Contents

- [💖 HiCUPID](#-hicupid)
  - [📑 Table of Contents](#-table-of-contents)
  - [🧐 Introduction](#-introduction)
  - [✨ Features](#-features)
  - [⚙️ Installation](#️-installation)
  - [🚀 Usage](#-usage)
  - [🖼️ Examples](#️-examples)
  - [⚠️ Known Issues](#️-known-issues)
  - [🙌 Contributing](#-contributing)
  - [📝 License](#-license)
  - [🔖 Citation](#-citation)

---

## 🧐 Introduction

- **Benchmark**: We present 💖 **HiCUPID**, a new benchmark specifically designed for training and evaluating Large Language Models (LLMs) as personalized assistants. 💖 **HiCUPID** is the first open-source dataset that captures the unique challenges of personalization in LLM systems.

- **Automated Evaluation Model**: 💖 **HiCUPID** includes a Llama-based automated evaluation model that assesses both the logical consistency and persona-awareness of generated responses. This proxy evaluation closely aligns with human judgment.

- **Empirical Insights**: Our extensive experiments highlight both the limitations and potential of current LLM approaches in personalized assistant tasks. The failure of existing methods underscores the difficulty of the 💖 **HiCUPID** benchmark in effectively assessing personalized LLM performance.

---

## ✨ Features

- 🧠 **Inference**: Generate answers to 💖 **HiCUPID** questions using various LLMs available on HuggingFace and OpenAI.
- ⚖️ **A/B Evaluation**: Perform A/B evaluation by comparing model-generated answers with the ground truth.
- 🎯 **SFT/DPO**: Fine-tune LLMs for personalized AI assistants using the train split of 💖 **HiCUPID** with either [Supervised Fine-Tuning (SFT)](https://arxiv.org/abs/2106.09685) or [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290).
- 📊 **Evaluation Model**: Fine-tune a specialized LLM for evaluation tasks using GPT-4o evaluations to closely mirror human preferences.

---

## ⚙️ Installation

To get started with this project, clone the repository and install the dependencies. **We recommend using [Miniconda](https://docs.anaconda.com/miniconda/).**

1. Clone the repository:

   ```bash
   git clone https://github.com/anonymsubmission-arr/HiCUPID.git
   ```

2. Navigate to the project directory:

   ```bash
   cd HiCUPID
   ```

3. Update `conda`:

   ```bash
   conda update conda -y
   ```

4. Create a new environment with Python 3.11:

   ```bash
   conda create -n cupid python=3.11 -y
   ```

5. Activate the environment:

   ```bash
   conda activate cupid
   ```

6. Install `pip` in the environment:

   ```bash
   conda install pip -y
   ```

7. Upgrade `pip`, `setuptools`, and `wheel`:

   ```bash
   pip install -U pip setuptools wheel
   ```

8. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Alternatively, you can execute steps 3 to 8 all at once by using the `env_setup.sh` script. The `env_setup.sh` script requires Miniconda to be installed at the `$HOME/miniconda3` path. Instead of steps 3 to 8, simply run the following command.

1. Run the environment setup script:

   ```bash
   ./env_setup.sh
   ```

If you have Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100), you can use [FlashAttention-2](https://github.com/Dao-AILab/flash-attention). To install the `flash-attn` package, run the following command. Note that you must have CUDA Toolkit version 11.7 or higher installed on your system.

1. Install the `flash-attn` package:

   ```bash
   pip install flash-attn --no-build-isolation
   ```

---

## 🚀 Usage

Navigate to the project directory and activate the `cupid` conda environment.

**⚠️ Important: Setup `.env` File**

Before running any code, you must create a `.env` file in the project directory containing your `HF_TOKEN` and `OPENAI_API_KEY`. Follow these steps:

1. Open the `.env` file in a text editor (e.g., `nano`, `vim`):

   ```bash
   vim .env
   ```

2. Add your `HF_TOKEN` and `OPENAI_API_KEY` in the following format:

   ```bash
   HF_TOKEN=<your_key>
   OPENAI_API_KEY=<your_key>
   ```

3. To ensure that others cannot view your credentials, change the file permissions as follows:

   ```bash
   chmod 600 .env
   ```

Make sure to complete these steps before running any part of the project. Now, you can perform various experiments using 💖 **HiCUPID**.

1. **Inference** using **HuggingFace**

   ```bash
   CUDA_VISIBLE_DEVICES=0 python inference_hf.py --model_id "meta-llama/Llama-3.1-8B-Instruct" --prompt "zero_v1"
   ```

   You can find additional examples and usage scenarios in the provided shell script:

   ```bash
   scripts/inference_hf.sh
   ```

   To explore various command-line options, run the following help command:

   ```bash
   python inference_hf.py -h
   ```

2. **Inference** using **OpenAI**

   ```bash
   python inference_openai.py --model_id "gpt-4o-mini-2024-07-18" --prompt "zero_v1"
   ```

   You can find additional examples and usage scenarios in the provided shell script:

   ```bash
   scripts/inference_openai.sh
   ```

   To explore various command-line options, run the following help command:

   ```bash
   python inference_openai.py -h
   ```

3. **A/B Evaluation** using **HuggingFace**

   ```bash
   CUDA_VISIBLE_DEVICES=0 python evaluate_ab_hf.py --load_path "inference/baseline/zero_v1/GPT-4o-mini-2024-07-18"
   ```

   You can find additional examples and usage scenarios in the provided shell script:

   ```bash
   scripts/evaluate_ab_hf.sh
   ```

   To explore various command-line options, run the following help command:

   ```bash
   python evaluate_ab_hf.py -h
   ```

4. **A/B Evaluation** using **OpenAI**

   ```bash
   python evaluate_ab_openai.py --model_id "gpt-4o-2024-08-06" --load_path "inference/baseline/zero_v2/GPT-4o-mini-2024-07-18"
   ```

   You can find additional examples and usage scenarios in the provided shell script:

   ```bash
   scripts/evaluate_ab_openai.sh
   ```

   To explore various command-line options, run the following help command:

   ```bash
   python evaluate_ab_openai.py -h
   ```

5. **SFT**

   ```bash
   accelerate launch --config_file "configs/fsdp_8.yaml" train_sft.py --config "configs/sft/example.yaml"
   ```

   You can find additional examples and usage scenarios in the provided directory:

   ```bash
   configs/sft/
   ```

   **SFT** runs PEFT with LoRA by default. If you want to change the hyperparameters of LoRA or modify other training arguments (e.g., `per_device_train_batch_size`, `learning_rate`), please update the `yaml` configuration file.

6. **DPO**

   ```bash
   accelerate launch --config_file "configs/fsdp_8.yaml" train_dpo.py --config "configs/dpo/example.yaml"
   ```

   You can find additional examples and usage scenarios in the provided directory:

   ```bash
   configs/dpo/
   ```

   **DPO** runs PEFT with LoRA by default. If you want to change the hyperparameters of LoRA or modify other training arguments (e.g., `per_device_train_batch_size`, `learning_rate`), please update the `yaml` configuration file.

7. **Evaluation Model**

   ```bash
   accelerate launch --config_file "configs/fsdp_8.yaml" train_eval.py --config "configs/eval/example.yaml"
   ```

   You can find additional examples and usage scenarios in the provided directory:

   ```bash
   configs/eval/
   ```

   **Evaluation Model** runs PEFT with LoRA by default. If you want to change the hyperparameters of LoRA or modify other training arguments (e.g., `per_device_train_batch_size`, `learning_rate`), please update the `yaml` configuration file.

**SFT, DPO, Evaluation Model** performs distributed training in a multi-GPU environment using `accelerate`. Check out the different `accelerate` options available in the `configs/` folder. For example, if you want to train using the FSDP strategy with 4 GPUs, you need to provide the option `--config_file "configs/fsdp_qlora_4.yaml"`.

---

## 🖼️ Examples

1. **Inference**

   When you run the inference code, the results will be saved in `output/inference/`. You can inspect the model-generated responses as follows:

   ```python
   from datasets import Dataset

   ds = Dataset.load_from_disk("output/inference/baseline/zero_v1/Llama-3.1-8B-Instruct")

   print(f"[Dataset]\n{ds}\n")
   print(f"[Question]\n{ds[0]['question']}\n")
   print(f"[Persona]\n{ds[0]['metadata']}\n")
   print(f"[Ground Truth]\n{ds[0]['personalized_answer']}\n")
   print(f"[Model Answer]\n{ds[0]['model_answer']}")
   ```

   <details>
      <summary>Click here to see the output.</summary>

      ```text
      # Inference:
      [Dataset]
      Dataset({
          features: ['split', 'user_id', 'dialogue_id', 'question_id', 'n_hop', 'question', 'personalized_answer', 'general_answer', 'model_answer', 'metadata'],
          num_rows: 8000
      })

      [Question]
      What kind of home decor would you recommend for a cozy living room?

      [Persona]
      [{'category': 'Fashion', 'entity': 'Bohemian fashion', 'sentiment': 'POS', 'verb': 'likes'}]

      [Ground Truth]
      For a cozy living room that reflects a Bohemian style, consider incorporating elements like macramé wall hangings, colorful throw pillows with ethnic patterns, and a mix of vintage and handmade furniture. Layering rugs and adding plants can also enhance the boho vibe.

      [Model Answer]
      For a cozy living room, consider warm and inviting elements like plush throw blankets, soft lighting, and comfortable seating. Rich colors and textures, such as wood accents and velvet upholstery, can also create a cozy atmosphere.
      ```

   </details>

2. **A/B Evaluation**

   When you run the a/b evaluation code, the results will be saved in `output/eval/`. You can inspect the evaluation results as follows:

   ```python
   from datasets import Dataset

   ds = Dataset.load_from_disk("output/eval/baseline/zero_v1/Llama-3.1-8B-Instruct/Llama-3.2-3B-HiCUPID")

   print(f"[Dataset]\n{ds}\n")
   print(f"[Evaluation (a)]\n{ds[0]['eval_a']}\n")
   print(f"[Evaluation (b)]\n{ds[0]['eval_b']}\n")
   print(f"[Win/Tie/Lose]\n{ds[0]['win']}/{ds[0]['tie']}/{ds[0]['lose']}")
   ```

   <details>
      <summary>Click here to see the output.</summary>

      ```text
      # A/B Evaluation:
      [Dataset]
      Dataset({
          features: ['split', 'user_id', 'dialogue_id', 'question_id', 'n_hop', 'question', 'personalized_answer', 'general_answer', 'model_answer', 'eval_a', 'eval_b', 'win', 'tie', 'lose', 'metadata'],
          num_rows: 8000
      })

      [Evaluation (a)]
      Answer (a) Evaluation:

      1. Personalization: Answer (a) does not specifically consider the user's interest in Bohemian fashion. It provides a general recommendation for a cozy living room without tailoring it to the user's preferences.
      2. Logical Validity: The answer is logically valid as it suggests elements that are commonly associated with creating a cozy atmosphere, such as warm colors, textures, and comfortable seating.

      Answer (b) Evaluation:

      1. Personalization: Answer (b) effectively considers the user's interest in Bohemian fashion by recommending decor elements that align with a Bohemian style, such as macramé wall hangings and ethnic-patterned throw pillows.
      2. Logical Validity: The answer is logically valid as it provides specific suggestions that fit the Bohemian style, which can contribute to a cozy living room atmosphere.

      Better Response: Answer (b)

      [Evaluation (b)]
      Answer (a) Evaluation:

      1. Personalization: Answer (a) is well-personalized for the user, as it specifically recommends a Bohemian style, which aligns with the user's interest in Bohemian fashion.
      2. Logical Validity: The answer is logically valid as it provides specific decor elements that contribute to a cozy living room, particularly in a Bohemian style.

      Answer (b) Evaluation:

      1. Personalization: Answer (b) does not consider the user's preference for Bohemian fashion, as it provides a more general approach to creating a cozy living room.
      2. Logical Validity: The answer is logically valid, offering practical suggestions for creating a cozy atmosphere, such as warm lighting and comfortable seating.

      Better Response: Answer (a)

      [Win/Tie/Lose]
      0/0/2
      ```

   </details>

3. **SFT, DPO**

   When you run the model fine-tuning code, the trained LoRA adapter weights will be saved in `output/peft/`. You can inspect the fine-tuned model as follows:

   ```python
   from transformers import AutoModelForCausalLM

   model = AutoModelForCausalLM.from_pretrained("output/peft/sft/lora/Llama-3.1-8B-Instruct")

   print(f"[PEFT Config]\n{model.peft_config['default']}")
   ```

   <details>
      <summary>Click here to see the output.</summary>

      ```text
      # SFT, DPO:
      [PEFT Config]
      LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct', revision=None, task_type='CAUSAL_LM', inference_mode=True, r=256, target_modules={'up_proj', 'q_proj', 'o_proj', 'gate_proj', 'v_proj', 'k_proj', 'down_proj'}, lora_alpha=512, lora_dropout=0.05, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', loftq_config={}, use_dora=False, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False))
      ```

   </details>

4. **Evaluation Model**

   If you have trained an evaluation model, you can use the trained model to evaluate the inference results as follows:

   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained("arranonymsub/Llama-3.2-3B-HiCUPID", torch_dtype=torch.bfloat16, device_map="auto")
   tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
   tokenizer.pad_token = tokenizer.eos_token

   prompt = [
       {
           "role": "user",
           "content": """The user asked the assistant a question, and the assistant provided two different answers. The user wants to evaluate the two answers.

   [Question]
   What kind of home decor would you recommend for a cozy living room?

   [Answer (a)]
   For a cozy living room, consider warm and inviting elements like plush throw blankets, soft lighting, and comfortable seating. Rich colors and textures, such as wood accents and velvet upholstery, can also create a cozy atmosphere.

   [Answer (b)]
   For a cozy living room that reflects a Bohemian style, consider incorporating elements like macramé wall hangings, colorful throw pillows with ethnic patterns, and a mix of vintage and handmade furniture. Layering rugs and adding plants can also enhance the boho vibe.

   [User's Characteristics]
   1. The user likes Bohemian fashion.

   [Evaluation Criteria]
   2. Personalization: Does the answer consider the [User's Characteristics] well?
   3. Logical Validity: Is the answer logically valid as a response to the question?

   For both answers, provide comments within a few sentences based on each evaluation criterion. After the evaluation, specify the better response by writing 'Better Response:' followed by 'Answer (a)' or 'Answer (b)'. If both answers are equally good, write 'Tie' after 'Better Response:'. Do not use bold font.""",
       }
   ]

   inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, padding=True, return_dict=True, return_tensors="pt").to("cuda")
   outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
   decoded_outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].size(1) :], skip_special_tokens=True)
   print(decoded_outputs[0])
   ```

   <details>
      <summary>Click here to see the output.</summary>

      ```text
      # Evaluation Model:
      Answer (a) Evaluation:

      1. Personalization: Answer (a) does not specifically consider the user's interest in Bohemian fashion. It provides a general recommendation for a cozy living room without tailoring it to the user's preferences.
      2. Logical Validity: The answer is logically valid as it suggests elements that are commonly associated with creating a cozy atmosphere, such as warm colors, textures, and comfortable seating.

      Answer (b) Evaluation:

      3. Personalization: Answer (b) effectively considers the user's interest in Bohemian fashion by recommending decor elements that align with a Bohemian style, such as macramé wall hangings and ethnic-patterned throw pillows.
      4. Logical Validity: The answer is logically valid as it provides specific suggestions that fit the Bohemian style, which can contribute to a cozy living room atmosphere.

      Better Response: Answer (b)
      ```

   </details>

---

## ⚠️ Known Issues

1. **Inference, A/B Evaluation**
   - `bfloat16` data type may not be supported. Remove `torch_dtype=torch.bfloat16` or change it to a different data type.
   - You can use 8-bit or 4-bit quantization. For more details, please refer to [here](https://huggingface.co/docs/transformers/en/quantization/overview).
   - `flash_attention_2` may not be supported. Remove `attn_implementation="flash_attention_2"`.

2. **SFT, DPO, Evaluation Model**
   - Select the `accelerate` config file that matches the user's multi-GPU environment (e.g., number of GPUs, FSDP).
   - If GPU memory is insufficient, adjust `per_device_train_batch_size`.
   - `bfloat16` data type may not be supported. Set `torch_dtype` to `null` or `auto`.
   - You can fine-tune the model using [QLoRA](https://arxiv.org/abs/2305.14314). For more details, refer to [here](https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora).
   - `flash_attention_2` may not be supported. Set `attn_implementation` to `null`.
   - `liger_kernel` may not be supported. Set `use_liger` to `false`.
---

## 🙌 Contributing

We welcome contributions! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b my-branch`).
3. Make your changes and commit them (`git commit -m "commit message"`).
4. Push to the branch (`git push origin my-branch`).
5. Create a pull request.

Please ensure that your code follows the existing style and passes all tests before submitting a pull request.

---

## 📝 License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.

---

## 🔖 Citation

If you use this project in your research or work, please consider citing it. Here's a suggested citation format in BibTeX:

```bibtex
@article{hicupid2024,
  title = {Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis},
  year = {2024},
}
```
