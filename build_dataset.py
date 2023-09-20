import json
from functools import partial

from datasets import Dataset
from loguru import logger
from transformers import AutoTokenizer


def create_prompt(sample):
    messages = sample["messages"]
    prompt = ""
    for message in messages:
        prompt += f"<{message['role']}> {message['name']}: \n"
        prompt += f"{message['content']}\n\n"

    return prompt


def get_max_length(model):
    max_length = None

    for length_setting in [
        "n_positions",
        "max_position_embeddings",
        "seq_length",
        "model_max_length",
    ]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max length: {max_length}")
            break

    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")

    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    logger.info("Preprocessing dataset...")

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "baichuan-inc/Baichuan2-13B-Base",
        use_auth_token=True,
        use_fast=False,
        trust_remote_code=True,
    )

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    with open("data-parser/data/qq-group-messages.jsonl") as f:
        dataset = [create_prompt(json.loads(line)) for line in f]
    logger.info(f"Loaded {len(dataset)} samples")

    dataset = Dataset.from_dict({"text": dataset})

    # 4096 is the max length of the model
    dataset = preprocess_dataset(tokenizer, 4096, 42, dataset)

    dataset.save_to_disk("data/qq-group-messages-tokenized")
