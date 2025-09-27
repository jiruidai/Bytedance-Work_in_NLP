import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
import gensim
import random
import gensim.downloader as api

from google.colab import drive
drive.mount('/content/drive')

# 加载 spaCy 中文模型
nlp = spacy.load("zh_core_web_sm")

# 读取数据集
data = pd.read_csv('/content/drive/MyDrive/douyin_sentiment.csv')

# 加载预训练的 Word2Vec 模型
# 你需要确保预先下载好的Word2Vec模型路径正确
w2v_model = api.load('word2vec-google-news-300')

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

# 数据增强：同义词替换
def synonym_replacement(text, prob=0.1):
    words = list(text)
    augmented_text = []
    for word in words:
        if random.random() < prob and word in w2v_model:
            # 替换为相似词
            similar_words = w2v_model.most_similar(word, topn=10)
            new_word = random.choice(similar_words)[0]
            augmented_text.append(new_word)
        else:
            augmented_text.append(word)
    return ''.join(augmented_text)

# 数据增强：对标注后的数据进行同义词替换
def augment_data_with_synonyms(data, prob=0.1):
    augmented_data = []
    for text in data['干净评论']:
        augmented_text = synonym_replacement(text, prob)
        augmented_data.append(augmented_text)
    return augmented_data

# 提取和标注评论
data['标注评论'] = data['干净评论'].apply(annotate_entities)

# 增加同义词替换后的数据增强
augmented_comments = augment_data_with_synonyms(data, prob=0.1)
data['增强评论'] = augmented_comments
data['增强标注评论'] = data['增强评论'].apply(annotate_entities)

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

# 打印一些统计信息
print("\n数据集统计：")
print(f"包含实体的句子总数：{len(formatted_annotations)}")
print(f"训练集大小：{len(train)}")
print(f"验证集大小：{len(val)}")
print(f"测试集大小：{len(test)}")

# 保存训练集、验证集和测试集到文件
save_to_file(train, 'douyin_train.txt')
save_to_file(val, 'douyin_val.txt')
save_to_file(test, 'douyin_test.txt')

from google.colab import files
files.download('douyin_train.txt')
files.download('douyin_val.txt')
files.download('douyin_test.txt')