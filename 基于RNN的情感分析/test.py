import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import build_model  # 引入模型结构

# 参数设置
max_len = 200
input_dim = 10000
output_dim = 128
weights_path = "gru_weights.h5"
tokenizer_path = "tokenizer.pkl"

# 加载分词器
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError("❌ 未找到 tokenizer.pkl，请先训练模型生成分词器")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
print("📦 已加载分词器")

# 构建模型
model = build_model(input_dim=input_dim, output_dim=output_dim, input_length=max_len)

# 加载模型参数
if not os.path.exists(weights_path):
    raise FileNotFoundError("❌ 未找到模型权重 gru_weights.h5，请先训练模型保存权重")
model.load_weights(weights_path)
print("✅ 模型已加载完毕，可以进行预测！")

# 进入预测循环
print("\n🎯 输入一段英文影评内容，模型将预测其情感倾向（输入 q 或 quit 退出）")
while True:
    user_input = input("请输入影评内容：").strip()
    if user_input.lower() in ['q', 'quit']:
        print("👋 退出测试")
        break

    # 文本预处理
    seq = tokenizer.texts_to_sequences([user_input])
    pad_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # 模型预测
    pred = model.predict(pad_seq)[0][0]
    label = int(pred >= 0.5)
    label_text = "积极" if label == 1 else "消极"

    print(f"📢 模型预测：{label_text}（概率值：{pred:.4f}）\n")
