# LLM-AVATAR

制作你的数字分身. 

## 1. 数据准备
目前支持的数据格式为:
- [x] QQ 群聊数据

运行以下命令, 生成数据集:

```bash
cd data-parser
cargo run --release --bin qq-group-messages -- --user-id 你的 QQ 号 --password "数据库密码"
```

以上命令会在 `data-parser/data/qq-group-messages.jsonl` 生成 OpenAI 格式的数据集, 我们需要将其转为 HuggingFace 格式:

```bash
python build_dataset.py
```

该命令会在 `data/qq-group-messages-tokenized` 生成 HuggingFace 格式的数据集.

## 2. 训练模型

```bash
bash finetune.sh
```

你可以调整 `finetune.sh` 中的参数, 来选择是否使用 lora.

如果你使用了 Lora, 在训练完成后可以使用 `merge.py` 来合并权重.

