import pandas as pd
from snownlp import SnowNLP

# 读取已经预处理过的评论数据
df = pd.read_csv('processed_douyin_data_large_balanced.csv')

# 由于之前数据预处理时就有情感标签列，先删除该列
if '情感标签' in df.columns:
    df = df.drop(columns=['情感标签'])

# 定义情感分析函数，基于之前已有的分词结果进行情感分析
def analyze_sentiment(segmented):
    # 将分词结果拼接成一个句子
    words = [word for word, pos in eval(segmented)]
    sentence = ' '.join(words)
    
    # 检查拼接后的句子是否为空
    if not sentence.strip():  
        return 0.5  # 如果句子为空，返回一个中性的情感得分 0.5
    
    # 使用SnowNLP库进行情感分析
    try:
        s = SnowNLP(sentence)
        return s.sentiments  # 返回情感得分，范围在[0, 1]
    except ZeroDivisionError:
        return 0.5  # 如果出现除零错误，返回中性情感得分 0.5

# 对每条评论的分词结果进行情感分析，并生成情感得分
df['情感得分'] = df['segmented'].apply(analyze_sentiment)

# 基于情感得分进行积极、消极二分类
df['情感标签'] = df['情感得分'].apply(lambda x: '积极' if x >= 0.5 else '消极')

# 统计积极和消极评论的数量
sentiment_counts = df['情感标签'].value_counts()

# 计算积极和消极评论的比例
positive_ratio = sentiment_counts.get('积极', 0) / len(df) * 100
negative_ratio = sentiment_counts.get('消极', 0) / len(df) * 100

# 输出结果
print(f"积极评论占比: {positive_ratio:.2f}%")
print(f"消极评论占比: {negative_ratio:.2f}%")

# 保存数据
df.to_csv('douyin_sentiment.csv', index=False, encoding='utf-8-sig')