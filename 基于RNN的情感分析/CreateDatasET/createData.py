from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import json

# 加载IMDB数据集
dataset = load_dataset("parquet", data_files="test-00000-of-00001.parquet")

# 提取训练集
train_data = dataset["train"]

# 将 train_data 转换为 pandas DataFrame
train_df = pd.DataFrame(train_data)

# 将训练集划分为70%训练集和30%测试集
train_split, test_split = train_test_split(
    train_df, test_size=0.3, random_state=42, stratify=train_df["label"]
)
# 保存函数
def save_dataset_to_json(dataset_split, filename):
    with open(filename, "w", encoding="utf-8") as f:
        # 转换成字典格式
        dataset_json = []
        for index, row in dataset_split.iterrows():
            # 将每个样本转换为字典，并添加到列表中
            dataset_json.append({"label": row['label'], "text": row['text'].replace('\n', ' ').strip()})
        
        # 将整个数据集保存为JSON文件
        json.dump(dataset_json, f, ensure_ascii=False, indent=4)

# 保存文件
save_dataset_to_json(train_split, "train.json")
save_dataset_to_json(test_split, "test.json")
print("数据集划分并保存成功 ✅")
