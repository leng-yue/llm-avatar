# Mostly copied from https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/
# which relies on https://github.com/databrickslabs/dolly/blob/master/training/trainer.py

import os

import bitsandbytes as bnb
import torch
from datasets import load_from_disk
from loguru import logger
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_model(model_name, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available resources
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, trust_remote_code=True
    )

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=True):
    """
    Prints the number of trainable parameters in the model.
    """

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params //= 2

    logger.info(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable: {100 * trainable_params / all_param:.2f} %"
    )


def train(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            learning_rate=2e-4,
            fp16=False,
            logging_steps=1,
            output_dir="logs",
            save_steps=100,
            save_total_limit=10,
            num_train_epochs=10,
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = (
        False  # re-enable for inference to speed up predictions for similar inputs
    )

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()

    total = 0
    for k, v in dtypes.items():
        total += v

    for k, v in dtypes.items():
        print(k, v, v / total)

    # Launch training
    logger.info("Training...")

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info(metrics)

    ###

    # Saving model
    logger.info("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


def merge_weights(output_dir):
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir, device_map="auto", torch_dtype=torch.bfloat16, rust_remote_code=True
    )
    model = model.merge_and_unload()

    output_merged_dir = output_dir + "/merged_checkpoint"
    os.makedirs(output_merged_dir, exist_ok=True)
    model.save_pretrained(output_merged_dir, safe_serialization=True)

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_merged_dir)


if __name__ == "__main__":
    # Load dataset
    dataset = load_from_disk("data/qq-group-messages-tokenized")

    logger.info(f"Loaded {len(dataset)} samples")

    # Load model from HF with user's token and with bitsandbytes config
    model_name = "baichuan-inc/Baichuan2-13B-Base"
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)

    output_dir = f"results/{model_name}"
    if os.path.exists(output_dir):
        logger.info(f"Checkpoint already exists at {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)

    train(model, tokenizer, dataset, output_dir)
    merge_weights(output_dir)
