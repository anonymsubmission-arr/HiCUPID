import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from itertools import chain

import tiktoken
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from openai.types import Batch
from openai.types.chat import ChatCompletion
from pandas import Series
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import *


def get_path(args):
    save_path = "/".join(["output", args.save_dir if args.save_dir is not None else "inference", "baseline", args.prompt, model_name(args.model_id)])
    return save_path


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
        tokens[feature + "_tokens"] = len(tokenizer.encode(example[feature]))
    return tokens


def count_tokens_from_messages(example, tokenizer):
    tokens = 0
    for message in example["messages"]:
        tokens += 3
        for k, v in message.items():
            tokens += len(tokenizer.encode(v))
            if k == "name":
                tokens += 1
    tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return {"messages_tokens": tokens}


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


def build_request(example, idx, **kwargs):
    request = {"custom_id": f"request-{idx}", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": example["messages"]} | kwargs}
    return {"request": request}


def save_input(dataset, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as file:
        for r in dataset["request"]:
            file.write(json.dumps(r) + "\n")


def save_batch(batch, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as file:
        file.write(batch.to_json())


def load_batch(fname):
    with open(fname, "r") as file:
        batch = Batch.model_validate_json(file.read())
    return batch


def wait_batch(client, batch_id, logger, sleep):
    i = 0
    while True:
        batch = client.batches.retrieve(batch_id)
        i += 1
        logger.info(f"Try {i}: {batch}")
        if batch.status == "completed":
            return batch
        elif batch.status == "in_progress":
            time.sleep(sleep)
            continue
        elif batch.status in ["validating", "finalizing", "cancelling"]:
            time.sleep(60)
            continue
        elif batch.status in ["failed", "expired", "cancelled"]:
            logger.error(f"Batch failed: {batch.status}")
            raise ValueError(f"Batch failed: {batch.status}")
        logger.error(f"Unseen batch status: {batch.status}")
        raise NotImplementedError(f"Unseen batch status: {batch.status}")


def save_output(output, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as file:
        file.write(output.text)
    return fname


def load_output(fname):
    with open(fname, "r") as file:
        requests = [json.loads(line) for line in file]
    requests = sorted(requests, key=lambda x: int(x["custom_id"].split("-")[1]))
    completions = [ChatCompletion.model_validate(r["response"]["body"]) for r in requests]
    return completions


def collate_fn(batch):
    return batch[0]["messages"]


def generate(rank, dataloader, client, logger, args):
    start = time.time()
    completions = []
    for i, messages in tqdm(enumerate(dataloader, start=1)):
        completion = client.chat.completions.create(messages=messages, model=args.model_id, max_completion_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
        completions.append(completion)
        if rank == 0 and i % args.log_every == 0:
            end = time.time()
            elapsed = end - start
            total = elapsed * (len(dataloader) / i)
            logger.info(f"Done {i}/{len(dataloader)} steps - {str(timedelta(seconds=int(elapsed)))}/{str(timedelta(seconds=int(total)))}.")
    return completions


def generate_batch(rank, dataset, client, logger, args):
    input_fname = args.save_path + f"/batch/input_{rank}.jsonl"
    batch_fname = args.save_path + f"/batch/batch_{rank}.json"
    output_fname = args.save_path + f"/batch/output_{rank}.jsonl"

    if args.cancel:
        batch = load_batch(batch_fname)
        client.batches.cancel(batch_id=batch.id)

    if not os.path.exists(batch_fname):
        dataset = dataset.map(partial(build_request, model=args.model_id, max_completion_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p), with_indices=True, num_proc=args.num_proc)
        save_input(dataset, input_fname)
        input = client.files.create(file=open(input_fname, "rb"), purpose="batch")
        batch = client.batches.create(input_file_id=input.id, endpoint="/v1/chat/completions", completion_window="24h")
        save_batch(batch, batch_fname)

    batch = load_batch(batch_fname)
    batch = wait_batch(client, batch.id, logger, args.sleep)
    output = client.files.content(batch.output_file_id)
    save_output(output, output_fname)
    completions = load_output(output_fname)
    return completions


def extract_completions(completions):
    contents = [completion.choices[0].message.content for completion in completions]
    prompt_tokens = [completion.usage.prompt_tokens for completion in completions]
    completion_tokens = [completion.usage.completion_tokens for completion in completions]
    return contents, prompt_tokens, completion_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HiCUPID inference using OpenAI models")
    parser.add_argument("--model_id", type=str, default="gpt-4o-mini", help="model name for inference")
    parser.add_argument("--dataset_id", type=str, default="arranonymsub/HiCUPID", help="dataset name for inference")
    parser.add_argument("--prompt", type=str, default="zero_v1", choices=PROMPT_MESSAGES.keys(), help="prompt method name")
    parser.add_argument("--max_tokens", type=int, default=300, help="(generation config) max new tokens")
    parser.add_argument("--temperature", type=float, default=0.5, help="(generation config) temperature")
    parser.add_argument("--top_p", type=float, default=0.5, help="(generation config) top p, nucleus sampling")
    parser.add_argument("--num_shards", type=int, default=16, help="number of shards for multithread processing")
    parser.add_argument("--num_proc", type=int, default=16, help="number of processors for processing datasets")
    parser.add_argument("--log_every", type=int, default=20, help="logging interval in steps")
    parser.add_argument("--sleep", type=int, default=180, help="time in seconds to sleep between batch retrievals")
    parser.add_argument("--cancel", action="store_true", default=False, help="cancel submitted batch jobs")
    parser.add_argument("--disable_batch", action="store_true", default=False, help="disable batch api calls")
    parser.add_argument("--split", type=str, default=None, help="which test split to use")
    parser.add_argument("--n_hop", type=int, default=None, help="which n_hop to use")
    parser.add_argument("--num_users", type=int, default=None, help="select only a subset of users")
    parser.add_argument("--save_dir", type=str, default=None, help="directory name to save outputs")
    args = parser.parse_args()
    args.save_path = get_path(args)

    logger = logging.getLogger("inference")
    logger.setLevel(logging.DEBUG)
    set_file_handler(logger, args.save_path)
    logger.info(f"Arguments: {args}")

    load_dotenv()
    login(token=os.environ["HF_TOKEN"])
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tokenizer = tiktoken.encoding_for_model(args.model_id)

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
    args.num_shards = min(len(dataset), args.num_shards)

    logger.info(f"question: {dataset[0]['question']}")
    logger.info(f"personalized_answer: {dataset[0]['personalized_answer']}")
    logger.info(f"general_answer: {dataset[0]['general_answer']}")
    logger.info(f"metadata: {dataset[0]['metadata']}")
    logger.info(f"question tokens (sum/mean/min/max): {statistics(dataset['question_tokens'])}")
    logger.info(f"personalized_answer tokens (sum/mean/min/max): {statistics(dataset['personalized_answer_tokens'])}")
    logger.info(f"general_answer tokens (sum/mean/min/max): {statistics(dataset['general_answer_tokens'])}")

    # Generate answer
    dataset = dataset.map(partial(count_tokens_from_messages, tokenizer=tokenizer), num_proc=args.num_proc)
    sharded_datasets = [dataset.shard(args.num_shards, i, contiguous=True) for i in range(args.num_shards)]

    logger.info(f"messages: {dataset[0]['messages']}")
    logger.info(f"messages tokens (sum/mean/min/max): {statistics(dataset['messages_tokens'])}")

    logger.info("Starting generation...")
    with ThreadPoolExecutor(max_workers=args.num_shards) as executor:
        if args.disable_batch:
            dataloaders = [DataLoader(ds, shuffle=False, collate_fn=collate_fn, pin_memory=True) for ds in sharded_datasets]  # type: ignore
            futures = [executor.submit(partial(generate, client=client, logger=logger, args=args), rank, dataloaders[rank]) for rank in range(args.num_shards)]
        else:
            futures = [executor.submit(partial(generate_batch, client=client, logger=logger, args=args), rank, sharded_datasets[rank]) for rank in range(args.num_shards)]
        results = [f.result() for f in futures]
        results = list(chain(*results))
        contents, prompt_tokens, completion_tokens = extract_completions(results)
    logger.info("Finished generation.")

    dataset = dataset.add_column("model_answer", contents)  # type: ignore

    logger.info(f"prompt tokens (sum/mean/min/max): {statistics(prompt_tokens)}")
    logger.info(f"completion tokens (sum/mean/min/max): {statistics(completion_tokens)}")

    # Save to disk
    dataset = dataset.select_columns(["split", "user_id", "dialogue_id", "question_id", "n_hop", "question", "personalized_answer", "general_answer", "model_answer", "metadata"])
    dataset.save_to_disk(args.save_path)

    logger.info(f"Saved dataset to {args.save_path}.")
