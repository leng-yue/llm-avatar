# LLM-AVATAR

制作你的数字分身. 

## 1. 数据准备
目前支持的数据格式为:
- [ ] QQ 群聊数据

运行以下命令, 生成数据集:

```bash
cd data-parser
cargo run --release --bin qq-group-messages -- --user-id 你的 QQ 号 --password "数据库密码"
```

