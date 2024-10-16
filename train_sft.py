import os
from dataclasses import dataclass, field
from functools import partial
from itertools import chain

from accelerate import PartialState
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from pandas import Series
from transformers import AutoTokenizer, set_seed
from trl import ModelConfig, SFTConfig, SFTScriptArguments, SFTTrainer, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config

from constants import *
from utils import DataCollatorForChatML


@dataclass
class Arguments:
    prompt: str = field(default="zero_v1", metadata={"help": "prompt method name"})
    train_seed: int = field(default=2024, metadata={"help": "random seed for train"})


def format_message(message, **kwargs):
    message["content"] = message["content"].format(**kwargs)
    return message


def build_dialogue(group):
    user = group["user"]
    assistant = group["assistant"]
    messages = [[{"role": "user", "content": u}, {"role": "assistant", "content": a}] for u, a in zip(user, assistant)]
    messages = list(chain(*messages))
    return Series({"messages": messages})


def build_messages(example, prompt, dialogue):
    dialogue = dialogue[example["user_id"]]["messages"]
    system = [m for m in prompt if m["role"] == "system"]
    user = [m for m in prompt if m["role"] != "system"]
    question = [{"role": "user", "content": example["question"]}]
    answer = [{"role": "assistant", "content": example["personalized_answer"]}]
    return {"messages": system + dialogue + user + question + answer}


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
        dialogue = load_dataset(script_args.dataset_name, name="dialogue", split="train")
        dialogue = Dataset.from_pandas(dialogue.to_pandas().groupby("user_id").apply(build_dialogue, include_groups=False))  # type: ignore
        dataset = load_dataset(script_args.dataset_name, name="qa", split="train")
        dataset = dataset.map(partial(build_messages, prompt=PROMPT_MESSAGES[args.prompt], dialogue=dialogue), num_proc=training_args.dataset_num_proc)  # type: ignore
        collator = DataCollatorForChatML(tokenizer=tokenizer, max_length=training_args.max_seq_length)

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,  # type: ignore
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
