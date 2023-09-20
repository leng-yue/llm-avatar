export NCCL_P2P_DISABLE=1

hostfile=""
deepspeed --hostfile=$hostfile fine-tune.py \
    --deepspeed ds_config.json \
    --report_to "tensorboard" \
    --data_path "data/qq-group-messages-tokenized" \
    --model_name_or_path "baichuan-inc/Baichuan2-7B-Base" \
    --output_dir "results" \
    --model_max_length 1024 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --use_lora True \
    --bf16 True \
    --tf32 True
