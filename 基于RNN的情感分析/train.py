import json
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import build_model  # 从 model.py 导入模型构建函数

# ------------------- 数据加载 -------------------
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels

train_texts, train_labels = load_data("train.json")
test_texts, test_labels = load_data("test.json")

# ------------------- 文本预处理 -------------------
# Tokenizer 文件路径
tokenizer_path = "tokenizer.pkl"

# 判断是否存在分词器
if os.path.exists(tokenizer_path):
    print("📦 加载已有分词器 tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
else:
    print("🛠️ 没有找到分词器，开始训练新的 Tokenizer")
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    
    # 保存分词器
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print("✅ 新分词器已保存到 tokenizer.pkl")

train_seqs = tokenizer.texts_to_sequences(train_texts)
test_seqs = tokenizer.texts_to_sequences(test_texts)

max_len = 200
train_pad = pad_sequences(train_seqs, maxlen=max_len, padding='post', truncating='post')
test_pad = pad_sequences(test_seqs, maxlen=max_len, padding='post', truncating='post')

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# ------------------- 模型准备 -------------------
weights_path = "gru_weights.h5"
model = build_model(input_dim=10000, output_dim=128, input_length=max_len)

if os.path.exists(weights_path):
    print("🔁 已检测到模型权重，加载中……")
    model.load_weights(weights_path)
else:
    print("🚀 未检测到模型权重，初始化新模型")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ------------------- 模型训练 -------------------
model.fit(train_pad, train_labels, epochs=5, batch_size=64, validation_data=(test_pad, test_labels))

# ------------------- 保存权重 -------------------
model.save_weights(weights_path)
print("✅ 模型训练完成，已保存权重到:", weights_path)
