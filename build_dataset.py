import json
from functools import partial

from datasets import Dataset
from loguru import logger
from transformers import AutoTokenizer


def preprocessing_function(
    sample,
    tokenizer,
    max_length,
    user_tokens=[195],
    assistant_tokens=[196],
    system_tokens=[197],
    ignore_index=-100,
    inference=False,
):
    input_ids = []
    labels = []

    messages = sample["text"]["messages"]
    for message in messages:
        content = f"{message['name']}: \n{message['content']}"
        value_ids = tokenizer.encode(content)

        if message["role"] == "user":
            input_ids += user_tokens + value_ids
            labels += [tokenizer.eos_token_id] + [ignore_index] * len(value_ids)
        elif message["role"] == "assistant":
            input_ids += assistant_tokens + value_ids
            labels += [ignore_index] + value_ids
        elif message["role"] == "system":
            input_ids += system_tokens + value_ids
            labels += [ignore_index] + value_ids

    if inference is False:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)

    if inference is False:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        labels += [ignore_index] * (max_length - len(labels))
        attention_mask += [0] * (max_length - len(attention_mask))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    logger.info("Preprocessing dataset...")

    dataset = dataset.map(
        partial(preprocessing_function, max_length=max_length, tokenizer=tokenizer),
    )

    # RM keys that are not needed
    dataset = dataset.remove_columns(["text"])

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "baichuan-inc/Baichuan2-7B-Base",
        use_fast=False,
        trust_remote_code=True,
    )

    # Needed for LLaMA tokenizer
    with open("data-parser/data/qq-group-messages.jsonl") as f:
        dataset = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(dataset)} samples")

    dataset = Dataset.from_dict({"text": dataset})

    # 1024 is the max length of the dataset, change this if you want to
    dataset = preprocess_dataset(tokenizer, 1024, 42, dataset)

    dataset.save_to_disk("data/qq-group-messages-tokenized")
