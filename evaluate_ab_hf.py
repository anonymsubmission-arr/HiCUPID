import argparse
import copy
import logging
import os
import re
import time
from datetime import timedelta
from functools import partial

import torch
from datasets import Dataset, disable_caching
from dotenv import load_dotenv
from huggingface_hub import login
from peft.auto import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import *


def get_path(args):
    model = args.peft_path.split("/")[-1:] if args.peft_path is not None else [model_name(args.model_id)]
    save_path = "/".join(["output", args.save_dir if args.save_dir is not None else "eval"] + args.load_path.split("/")[1:] + model)
    load_path = "/".join(["output"] + args.load_path.split("/"))
    peft_path = "/".join(["output"] + args.peft_path.split("/")) if args.peft_path is not None else None
    return save_path, load_path, peft_path


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


def build_messages(batch, messages):
    messages = copy.deepcopy(messages)
    example = {k: v[0] for k, v in batch.items()}
    metadata = [METADATA.format(idx=i, verb=m["verb"], entity=m["entity"]) for i, m in enumerate(example["metadata"], start=1)]
    metadata = "\n".join(metadata)
    messages_a = [format_message(m, **(example | {"metadata": metadata})) for m in messages["a"]]
    messages_b = [format_message(m, **(example | {"metadata": metadata})) for m in messages["b"]]
    return {k: v * 2 for k, v in batch.items()} | {"messages": [messages_a, messages_b]}


def add_messages(example, messages):
    messages = copy.deepcopy(messages)
    messages = [format_message(m, **example) for m in messages]
    return {"messages": example["messages"] + messages}


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


def parse_evaluation(eval, label, logger):
    pattern = re.compile(r"Better Response:.*(?:Answer \((a|b)\)|Tie)", re.IGNORECASE)
    match = pattern.search(eval)
    if match is None:
        logger.error(f"Match has not been found: {eval}")
        return "tie"
    if match.group(1) is None:
        return "tie"
    elif match.group(1).lower() == label:
        return "win"
    else:
        return "lose"


def score(batch, logger):
    example = {k: v[0] for k, v in batch.items()}
    eval_a, eval_b = batch["eval"]
    score = {"win": 0, "tie": 0, "lose": 0}
    score[parse_evaluation(eval_a, "a", logger)] += 1
    score[parse_evaluation(eval_b, "b", logger)] += 1
    return example | {"eval_a": eval_a, "eval_b": eval_b} | score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HiCUPID evaluation using HuggingFace models")
    parser.add_argument("--model_id", type=str, default="arranonymsub/Llama-3.2-3B-HiCUPID", help="model name for evaluation")
    parser.add_argument("--peft_path", type=str, default=None, help="path to local peft adapter (e.g. peft/sft/lora/Llama-3.1-8B-Instruct)")
    parser.add_argument("--load_path", type=str, default="inference/baseline/zero_v1/Llama-3.1-8B-Instruct", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--max_tokens", type=int, default=300, help="generation config; max new tokens")
    parser.add_argument("--do_sample", type=bool, default=False, help="generation config; whether to do sampling, greedy if not set")
    parser.add_argument("--temperature", type=float, default=0.0, help="generation config; temperature")
    parser.add_argument("--top_k", type=int, default=50, help="generation config; top k")
    parser.add_argument("--top_p", type=float, default=0.1, help="generation config; top p, nucleus sampling")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--num_proc", type=int, default=16, help="number of processors for processing datasets")
    parser.add_argument("--log_every", type=int, default=20, help="logging interval in steps")
    parser.add_argument("--split", type=str, default=None, help="which test split to use")
    parser.add_argument("--n_hop", type=int, default=None, help="which n_hop to use")
    parser.add_argument("--num_users", type=int, default=None, help="select only a subset of users")
    parser.add_argument("--save_dir", type=str, default=None, help="directory name to save outputs")
    args = parser.parse_args()
    args.save_path, args.load_path, args.peft_path = get_path(args)

    logger = logging.getLogger("evaluate")
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
    disable_caching()

    # Preprocess
    dataset: Dataset = Dataset.load_from_disk(args.load_path)
    dataset = dataset.filter(lambda example: example["split"] == args.split, num_proc=args.num_proc) if args.split is not None else dataset
    dataset = dataset.filter(lambda example: example["user_id"] < args.num_users, num_proc=args.num_proc) if args.num_users is not None else dataset
    dataset = dataset.filter(lambda example: example["n_hop"] == args.n_hop, num_proc=args.num_proc) if args.n_hop is not None else dataset
    dataset = dataset.map(partial(build_messages, messages=EVAL_MESSAGES), batched=True, batch_size=1, remove_columns=dataset.column_names, num_proc=args.num_proc)
    dataset = dataset.map(partial(count_tokens, tokenizer=tokenizer, features=["question", "personalized_answer", "general_answer", "model_answer"]), num_proc=args.num_proc)

    logger.info(f"Loaded dataset from {args.load_path}.")
    logger.info(f"question: {dataset[0]['question']}")
    logger.info(f"personalized_answer: {dataset[0]['personalized_answer']}")
    logger.info(f"general_answer: {dataset[0]['general_answer']}")
    logger.info(f"model_answer: {dataset[0]['model_answer']}")
    logger.info(f"metadata: {dataset[0]['metadata']}")
    logger.info(f"question tokens (sum/mean/min/max): {statistics(dataset['question_tokens'])}")
    logger.info(f"personalized_answer tokens (sum/mean/min/max): {statistics(dataset['personalized_answer_tokens'])}")
    logger.info(f"general_answer tokens (sum/mean/min/max): {statistics(dataset['general_answer_tokens'])}")
    logger.info(f"model_answer tokens (sum/mean/min/max): {statistics(dataset['model_answer_tokens'])}")

    # Generate eval
    dataset = dataset.map(partial(build_prompt, tokenizer=tokenizer), num_proc=args.num_proc)

    logger.info(f"prompt: {dataset[0]['prompt']}")
    logger.info(f"prompt tokens (sum/mean/min/max): {statistics(dataset['prompt_tokens'])}")

    logger.info("Starting generation...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_proc, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True)  # type: ignore
    output_ids = generate(model, tokenizer, dataloader, logger, args.log_every, max_new_tokens=args.max_tokens, do_sample=args.do_sample, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    logger.info("Finished generation.")

    dataset = dataset.add_column("eval_ids", output_ids)  # type: ignore
    dataset = dataset.map(partial(decode, tokenizer=tokenizer, feature="eval"), num_proc=args.num_proc)

    # Score eval
    dataset = dataset.batch(batch_size=2)
    dataset = dataset.map(partial(score, logger=logger), num_proc=args.num_proc)

    logger.info(f"W/D/L: {sum(dataset['win']) / (2 * len(dataset)) * 100:.1f}/{sum(dataset['tie']) / (2 * len(dataset)) * 100:.1f}/{sum(dataset['lose']) / (2 * len(dataset)) * 100:.1f}")

    # Save to disk
    dataset = dataset.select_columns(["split", "user_id", "dialogue_id", "question_id", "n_hop", "question", "personalized_answer", "general_answer", "model_answer", "metadata", "eval_a", "eval_b", "win", "tie", "lose"])
    dataset.save_to_disk(args.save_path)

    logger.info(f"Saved dataset to {args.save_path}.")
