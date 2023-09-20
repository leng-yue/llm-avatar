import os

import torch
from huggingface_hub import snapshot_download
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_name = "baichuan-inc/Baichuan2-7B-Base"
output_dir = "results/checkpoint-245"
snapshot_download(model_name, local_dir=output_dir)

model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model = model.merge_and_unload()

output_merged_dir = output_dir + "/merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, use_fast=False
)
tokenizer.save_pretrained(output_merged_dir)
