import os
import warnings
from dataclasses import dataclass, field
from functools import partial

from accelerate import PartialState
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from trl import ModelConfig, RewardConfig, RewardTrainer, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config, maybe_apply_chat_template, setup_chat_format
from trl.commands.cli_utils import RewardScriptArguments

from constants import *
from utils import tokenize


@dataclass
class Arguments:
    margin: float | None = field(default=None, metadata={"help": "loss margin between chosen and rejected"})
    train_seed: int = field(default=2024, metadata={"help": "random seed for train"})


def build_messages(example):
    metadata = [METADATA.format(idx=i, verb=m["verb"], entity=m["entity"]) for i, m in enumerate(example["metadata"], start=1)]
    metadata = [{"role": "system", "content": "\n".join(metadata)}]
    question = [{"role": "user", "content": example["question"]}]
    chosen = [{"role": "assistant", "content": example["personalized_answer"]}]
    rejected = [{"role": "assistant", "content": example["general_answer"]}]
    return {"chosen": metadata + question + chosen, "rejected": metadata + question + rejected}


def main():
    parser = TrlParser((Arguments, RewardScriptArguments, RewardConfig, ModelConfig))
    args, script_args, training_args, model_config = parser.parse_args_and_config()  # type: ignore

    load_dotenv()
    login(token=os.environ["HF_TOKEN"])

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_config.model_name_or_path, num_labels=1, trust_remote_code=model_config.trust_remote_code, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)  # type: ignore
    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn("You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs" " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.")

    with PartialState().local_main_process_first():
        dataset = load_dataset(script_args.dataset_name, name="qa", split="train")  # type: ignore
        dataset = dataset.map(build_messages, num_proc=training_args.dataset_num_proc)  # type: ignore
        dataset = dataset.map(partial(maybe_apply_chat_template, tokenizer=tokenizer), num_proc=training_args.dataset_num_proc)  # type: ignore
        dataset = dataset.map(partial(tokenize, tokenizer=tokenizer), batched=True, num_proc=training_args.dataset_num_proc)
        dataset = dataset.filter(lambda x: len(x["input_ids_chosen"]) <= training_args.max_length and len(x["input_ids_rejected"]) <= training_args.max_length, num_proc=training_args.dataset_num_proc)
        dataset = dataset.map(lambda x: {"margin": args.margin}, num_proc=training_args.dataset_num_proc) if args.margin is not None else dataset  # type: ignore

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,  # type: ignore
        peft_config=get_peft_config(model_config),  # type: ignore
    )

    trainer.accelerator.print(f"{dataset[0]['chosen']}\n{dataset[0]['rejected']}")
    trainer.accelerator.print(f"{trainer.model}")
    trainer.accelerator.print(f"{trainer.args}")
    if model_config.use_peft:
        trainer.accelerator.print(f"{trainer.model.peft_config}")
        trainer.model.print_trainable_parameters()
        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy

            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)  # type: ignore

    set_seed(seed=args.train_seed)  # type: ignore
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)  # type: ignore

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")  # type: ignore
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)  # type: ignore


if __name__ == "__main__":
    main()
