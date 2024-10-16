import copy
import os
from dataclasses import dataclass, field
from functools import partial

from accelerate import PartialState
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, set_seed
from trl import ModelConfig, SFTConfig, SFTScriptArguments, SFTTrainer, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config

from constants import *
from utils import DataCollatorForChatML


@dataclass
class Arguments:
    train_seed: int = field(default=2024, metadata={"help": "random seed for train"})


def format_message(message, **kwargs):
    message["content"] = message["content"].format(**kwargs)
    return message


def build_messages(batch, messages):
    messages = copy.deepcopy(messages)
    example = {k: v[0] for k, v in batch.items()}
    metadata = [METADATA.format(idx=i, verb=m["verb"], entity=m["entity"]) for i, m in enumerate(example["metadata"], start=1)]
    metadata = "\n".join(metadata)
    messages_a = [format_message(m, **(example | {"metadata": metadata})) for m in messages["a"]] + [{"role": "assistant", "content": example["eval_a"]}]
    messages_b = [format_message(m, **(example | {"metadata": metadata})) for m in messages["b"]] + [{"role": "assistant", "content": example["eval_b"]}]
    return {k: v * 2 for k, v in batch.items() if k not in ["eval_a", "eval_b", "win", "tie", "lose"]} | {"messages": [messages_a, messages_b]}


def main():
    parser = TrlParser((Arguments, SFTScriptArguments, SFTConfig, ModelConfig))
    args, script_args, training_args, model_config = parser.parse_args_and_config()  # type: ignore

    load_dotenv()
    login(token=os.environ["HF_TOKEN"])

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token

    with PartialState().local_main_process_first():
        dataset = load_dataset(script_args.dataset_name, name="evaluation")
        dataset = concatenate_datasets([v.add_column("split", [k] * len(v)) for k, v in dataset.items()])  # type: ignore
        dataset = dataset.map(partial(build_messages, messages=EVAL_MESSAGES), batched=True, batch_size=1, remove_columns=dataset.column_names, num_proc=training_args.dataset_num_proc)
        collator = DataCollatorForChatML(tokenizer=tokenizer, max_length=training_args.max_seq_length)

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.accelerator.print(f"{dataset[0]['messages']}")
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
