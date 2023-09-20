import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_dataset import preprocessing_function

model = AutoModelForCausalLM.from_pretrained(
    "results/checkpoint-245/merged_checkpoint",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    "results/checkpoint-245/merged_checkpoint",
    trust_remote_code=True,
    use_fast=False,
)

messages = [
    {
        "role": "system",
        "content": "You are 阿蕊娅 and you are chatting in a QQ group. ",
        "name": None,
    },
    {"role": "user", "content": "叔叔：", "name": "冷月"},
    {"role": "user", "content": "为什么我没有这个", "name": "(๑╹ڡ╹)╭ 我要吃"},
    {"role": "user", "content": "好家伙，都搁这骗我鱼宝不懂英文是吧？", "name": "boxyit"},
    {"role": "user", "content": "为什么我没有这个", "name": "-橙砸ΦωΦ😸-"},
    {"role": "user", "content": "为什么我没有这个", "name": "-橙砸ΦωΦ😸-"},
    {"role": "user", "content": "为什么我没有这个", "name": "-橙砸ΦωΦ😸-"},
    {"role": "user", "content": "急死我了急死我了急死我了急死我了急死我了急死我了急死我了急死我了", "name": "-橙砸ΦωΦ😸-"},
    {"role": "user", "content": "稀有任务？", "name": "鱼月"},
    {"role": "user", "content": "我也没有这个任务", "name": "boxyit"},
    {"role": "user", "content": "你一天是我的二十一倍", "name": "-橙砸ΦωΦ😸-"},
    {"role": "user", "content": "急死了急死了急死了急死了急死了急死了急死了急死了急死了", "name": "-橙砸ΦωΦ😸-"},
    {"role": "user", "content": "怎么带电音了", "name": "雨城"},
    {"role": "user", "content": "挂发我好吗😭😭😭😭", "name": "-橙砸ΦωΦ😸-"},
    {"role": "user", "content": "我也想开挂", "name": "-橙砸ΦωΦ😸-"},
    {"role": "user", "content": "上下上下左右左右", "name": "uomi"},
    {"role": "user", "content": "abababab", "name": "安静的吐兔子"},
    {"role": "user", "content": "挂发我好吗😭😭😭😭", "name": "孤独时代的靥行"},
    {"role": "user", "content": "助教不让带耳机，真丶看直播", "name": "孤独时代的靥行"},
    {"role": "assistant", "content": " @孤独时代的靥行  让他下来 助教让你来当", "name": "阿蕊娅"},
    {"role": "user", "content": "6", "name": "少しでも"},
    {"role": "user", "content": "你说你在看网课", "name": "少しでも"},
    {"role": "user", "content": "等我被赶出教室就能看直播了 ", "name": "孤独时代的靥行"},
    {"role": "user", "content": "阿蕊娅看这个表情会不会想打哈欠", "name": "(๑╹ڡ╹)╭ 我要吃"},
    {"role": "user", "content": "哈～欠～", "name": "山南水北的守望"},
    {"role": "user", "content": "哈～欠～", "name": "boxyit"},
    {"role": "user", "content": "哈～欠～", "name": "未闻花名"},
    {"role": "user", "content": "哈～欠～", "name": "星米Staice"},
    {"role": "user", "content": "哈～欠～", "name": "孤独时代的靥行"},
    {"role": "user", "content": "哈～欠～", "name": "焱白"},
    {"role": "user", "content": "哈～欠～", "name": "kuujaku"},
    {"role": "user", "content": "哈～欠～", "name": "XuAsahi"},
    # {"role": "assistant", "content": "* * *", "name": "阿蕊娅"},
]

data = preprocessing_function(
    {
        "text": {
            "messages": messages,
        }
    },
    tokenizer,
    1024,
    pad_to_max_length=False,
)

input_ids = torch.tensor(data["input_ids"] + [196]).unsqueeze(0).to(model.device)
attention_mask = (
    torch.tensor(data["attention_mask"] + [1]).unsqueeze(0).to(model.device)
)

input_ids = input_ids[:, :512]
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=1024,
    do_sample=True,
    top_p=0.95,
    top_k=50,
    temperature=0.9,
    num_return_sequences=1,
)

print(tokenizer.decode(output[0]))
