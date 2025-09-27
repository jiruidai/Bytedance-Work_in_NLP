import pandas as pd
from sklearn.model_selection import train_test_split
import spacy

from google.colab import drive
drive.mount('/content/drive')


# 加载 spaCy 中文模型
nlp = spacy.load("zh_core_web_sm")

# 读取数据集
data = pd.read_csv('/content/drive/MyDrive/douyin_sentiment.csv')

# 进行命名实体标注
def annotate_entities(text):
    doc = nlp(text)
    annotations = []

    # 获取所有实体
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]

    # 处理每个字符
    for i, char in enumerate(text):
        tag = 'O'
        # 检查当前字符是否在任何实体中
        for start, end, label in entities:
            if start <= i < end:
                # 使用BIOES标签
                if start == i:  # 实体开始
                    if end - start == 1:  # 单字符实体
                        tag = f'S-{label}'
                    else:
                        tag = f'B-{label}'
                elif i == end - 1:  # 实体结束
                    tag = f'E-{label}'
                else:  # 实体中间
                    tag = f'I-{label}'
                break

        annotations.append((char, tag))

    return annotations

# 提取和标注评论
data['标注评论'] = data['干净评论'].apply(annotate_entities)

# 将标注数据转换为 (token, tag) 的元组列表
def format_annotated_data(annotations):
    formatted_data = []
    for sentence in annotations:
        # 只保存包含实体的句子
        if any(tag != 'O' for _, tag in sentence):
            formatted_data.append(sentence)
    return formatted_data

# 获取格式化的标注数据
formatted_annotations = format_annotated_data(data['标注评论'])

# 创建新的DataFrame
formatted_data_df = pd.DataFrame({'标注评论': formatted_annotations})

# 数据集划分，80%训练集、10%验证集、10%测试集
train, temp = train_test_split(formatted_data_df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# 保存到文件
def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for sentence in data['标注评论']:
            # 每个字符和标签占一行，句子之间用空行分隔
            for char, label in sentence:
                f.write(f"{char}\t{label}\n")
            f.write("\n")

# 保存训练集、验证集和测试集到文件
save_to_file(train, 'douyin_train.txt')
save_to_file(val, 'douyin_val.txt')
save_to_file(test, 'douyin_test.txt')

from google.colab import files
files.download('douyin_train.txt')
files.download('douyin_val.txt')
files.download('douyin_test.txt')