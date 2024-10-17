import argparse
import logging
import os
import time
from datetime import timedelta
from functools import partial
from itertools import chain

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from pandas import Series
from peft.auto import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import *


def get_path(args):
    method = args.peft_path.split("/")[1:] if args.peft_path is not None else ["baseline", args.prompt, model_name(args.model_id)]
    save_path = "/".join(["output", args.save_dir if args.save_dir is not None else "inference"] + method)
    peft_path = "/".join(["output"] + args.peft_path.split("/")) if args.peft_path is not None else None
    return save_path, peft_path


def set_file_handler(logger, path, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"):
    os.makedirs(os.path.dirname(path + "/run.log"), exist_ok=True)
    handler = logging.FileHandler(path + "/run.log")
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def count_tokens(example, tokenizer, features):
    tokens = {}
    for feature in features:
        tokens[feature + "_tokens"] = len(tokenizer(example[feature], add_special_tokens=False).input_ids)
    return tokens


def statistics(tokens):
    sum_t = sum(tokens)
    mean_t = round(sum_t / len(tokens))
    min_t = min(tokens)
    max_t = max(tokens)
    return f"{sum_t}/{mean_t}/{min_t}/{max_t}"


def format_message(message, **kwargs):
    message["content"] = message["content"].format(**kwargs)
    return message


def build_dialogue(group):
    user = group["user"]
    assistant = group["assistant"]
    messages = [[{"role": "user", "content": u}, {"role": "assistant", "content": a}] for u, a in zip(user, assistant)]
    messages = list(chain(*messages))
    return Series({"messages": messages})


def build_messages(example, prompt, dialogue, split):
    dialogue = dialogue[split][example["user_id"] - 500 if split == "test" else example["user_id"]]["messages"]
    system = [m for m in prompt if m["role"] == "system"]
    user = [m for m in prompt if m["role"] != "system"]
    question = [{"role": "user", "content": example["question"]}]
    return {"messages": system + dialogue + user + question}


def build_prompt(example, tokenizer):
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return {"prompt": prompt, "prompt_tokens": prompt_tokens}


def collate_fn(batch, tokenizer):
    prompt = [example["prompt"] for example in batch]
    inputs = tokenizer(prompt, add_special_tokens=False, padding=True, return_tensors="pt")
    return inputs


def generate(model, tokenizer, dataloader, logger, log_every, **kwargs):
    start = time.time()
    output_ids = []
    for i, inputs in tqdm(enumerate(dataloader, start=1)):
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **kwargs)
        output_ids.extend(outputs[:, inputs["input_ids"].size(1) :].tolist())
        if i % log_every == 0:
            end = time.time()
            elapsed = end - start
            total = elapsed * (len(dataloader) / i)
            logger.info(f"Done {i}/{len(dataloader)} steps - {str(timedelta(seconds=int(elapsed)))}/{str(timedelta(seconds=int(total)))}.")
    return output_ids


def decode(example, tokenizer, feature):
    text = tokenizer.decode(example[feature + "_ids"], skip_special_tokens=True)
    return {feature: text}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HiCUPID inference using HuggingFace models")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="model name for inference")
    parser.add_argument("--peft_path", type=str, default=None, help="path to local peft adapter (e.g. peft/sft/lora/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset_id", type=str, default="arranonymsub/HiCUPID", help="dataset name for inference")
    parser.add_argument("--prompt", type=str, default="zero_v1", choices=PROMPT_MESSAGES.keys(), help="prompt method name")
    parser.add_argument("--max_tokens", type=int, default=300, help="(generation config) max new tokens")
    parser.add_argument("--do_sample", type=bool, default=True, help="(generation config) whether to do sampling, greedy if not set")
    parser.add_argument("--temperature", type=float, default=0.5, help="(generation config) temperature")
    parser.add_argument("--top_k", type=int, default=50, help="(generation config) top k")
    parser.add_argument("--top_p", type=float, default=0.5, help="(generation config) top p, nucleus sampling")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for inference")
    parser.add_argument("--num_proc", type=int, default=16, help="number of processors for processing datasets")
    parser.add_argument("--log_every", type=int, default=20, help="logging interval in steps")
    parser.add_argument("--split", type=str, default=None, help="which test split to use")
    parser.add_argument("--n_hop", type=int, default=None, help="which n_hop to use")
    parser.add_argument("--num_users", type=int, default=None, help="select only a subset of users")
    parser.add_argument("--save_dir", type=str, default=None, help="directory name to save outputs")
    args = parser.parse_args()
    args.save_path, args.peft_path = get_path(args)

    logger = logging.getLogger("inference")
    logger.setLevel(logging.DEBUG)
    set_file_handler(logger, args.save_path)
    logger.info(f"Arguments: {args}")

    load_dotenv()
    login(token=os.environ["HF_TOKEN"])
    torch.set_float32_matmul_precision("high")
    if args.peft_path is not None:
        model = AutoPeftModelForCausalLM.from_pretrained(args.peft_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    # Preprocess
    dialogue: DatasetDict = load_dataset(args.dataset_id, name="dialogue")  # type: ignore
    dialogue = DatasetDict({k: v.filter(lambda example: example["user_id"] < (500 if k == "test" else 0) + args.num_users, num_proc=args.num_proc) for k, v in dialogue.items()}) if args.num_users is not None else dialogue
    dialogue = DatasetDict({k: Dataset.from_pandas(v.to_pandas().groupby("user_id").apply(build_dialogue, include_groups=False)) for k, v in dialogue.items()})

    qa: DatasetDict = load_dataset(args.dataset_id, name="qa")  # type: ignore
    qa = DatasetDict({k: v for k, v in qa.items() if k == args.split}) if args.split is not None else DatasetDict({k: v for k, v in qa.items() if k in ["test_1", "test_2"]})
    qa = DatasetDict({k: v.filter(lambda example: example["user_id"] < (500 if k == "test_2" else 0) + args.num_users, num_proc=args.num_proc) for k, v in qa.items()}) if args.num_users is not None else qa
    qa = qa.filter(lambda example: example["n_hop"] == args.n_hop, num_proc=args.num_proc) if args.n_hop is not None else qa
    qa = DatasetDict({k: v.map(partial(build_messages, prompt=PROMPT_MESSAGES[args.prompt], dialogue=dialogue, split=DIALOGUE_SPLIT[k]), num_proc=args.num_proc) for k, v in qa.items()})
    dataset: Dataset = concatenate_datasets([v.map(lambda example: {"split": k}) for k, v in qa.items()])
    dataset = dataset.map(partial(count_tokens, tokenizer=tokenizer, features=["question", "personalized_answer", "general_answer"]), num_proc=args.num_proc)

    logger.info(f"question: {dataset[0]['question']}")
    logger.info(f"personalized_answer: {dataset[0]['personalized_answer']}")
    logger.info(f"general_answer: {dataset[0]['general_answer']}")
    logger.info(f"metadata: {dataset[0]['metadata']}")
    logger.info(f"question tokens (sum/mean/min/max): {statistics(dataset['question_tokens'])}")
    logger.info(f"personalized_answer tokens (sum/mean/min/max): {statistics(dataset['personalized_answer_tokens'])}")
    logger.info(f"general_answer tokens (sum/mean/min/max): {statistics(dataset['general_answer_tokens'])}")

    # Generate answer
    dataset = dataset.map(partial(build_prompt, tokenizer=tokenizer), num_proc=args.num_proc)

    logger.info(f"prompt: {dataset[0]['prompt']}")
    logger.info(f"prompt tokens (sum/mean/min/max): {statistics(dataset['prompt_tokens'])}")

    logger.info("Starting generation...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_proc, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True)  # type: ignore
    output_ids = generate(model, tokenizer, dataloader, logger, args.log_every, max_new_tokens=args.max_tokens, do_sample=args.do_sample, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    logger.info("Finished generation.")

    dataset = dataset.add_column("model_answer_ids", output_ids)  # type: ignore
    dataset = dataset.map(partial(decode, tokenizer=tokenizer, feature="model_answer"), num_proc=args.num_proc)

    # Save to disk
    dataset = dataset.select_columns(["split", "user_id", "dialogue_id", "question_id", "n_hop", "question", "personalized_answer", "general_answer", "model_answer", "metadata"])
    dataset.save_to_disk(args.save_path)

    logger.info(f"Saved dataset to {args.save_path}.")
