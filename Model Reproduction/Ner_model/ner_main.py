import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from torchcrf import CRF
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

# 一、定义数据加载器:加载标注好的数据集并将其转换为模型可以处理的格式

# 定义NER数据集类
class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, tag2id, max_len=128):
        self.sentences, self.labels = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len

    def load_data(self, file_path):
        sentences, labels = [], []
        current_sentence = []
        current_labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 空行表示句子结束
                    if current_sentence:  # 如果当前句子不为空，保存它
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                    continue

                parts = line.split('\t')
                if len(parts) == 2:  # 确保行包含字符和标签
                    char, label = parts
                    current_sentence.append(char)
                    current_labels.append(label)

        # 处理最后一个句子
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)

        return sentences, labels

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        label_ids = [self.tag2id.get(tag, self.tag2id['O']) for tag in labels]
        if len(label_ids) > self.max_len:
            label_ids = label_ids[:self.max_len]
        elif len(label_ids) < self.max_len:
            label_ids.extend([self.tag2id['O']] * (self.max_len - len(label_ids)))

        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), torch.tensor(label_ids)

# 加载数据集
def load_datasets(train_path, val_path, test_path, tokenizer, tag2id, batch_size):
    train_dataset = NERDataset(train_path, tokenizer, tag2id)
    val_dataset = NERDataset(val_path, tokenizer, tag2id)
    test_dataset = NERDataset(test_path, tokenizer, tag2id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# 二、上下文编码模块，使用ZEN模型
# 初始化ZEN模型和分词器
from huggingface_hub import login
login(token="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")
tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-ZEN2-668M-Chinese')
zen_model = BertModel.from_pretrained('IDEA-CCNL/Erlangshen-ZEN2-668M-Chinese')

# 上下文编码函数
def encode_sentence(sentence):
    encoding = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = zen_model(**encoding)
    return outputs.last_hidden_state  # 获取最后一层的隐藏状态

# 三、语义增强模块:将基于预训练词嵌入提取每个词的最相似词，并通过注意力机制为其分配权重
def get_similar_words(word_vec, embedding_matrix, top_k=10):
    # 确保维度匹配
    if len(word_vec.shape) == 1:
        word_vec = word_vec.unsqueeze(0)  # [1, hidden_size]

    # 添加投影层来处理维度不匹配
    if word_vec.size(-1) != embedding_matrix.size(-1):
        projection = nn.Linear(word_vec.size(-1), embedding_matrix.size(-1))
        word_vec = projection(word_vec)

    # 计算余弦相似度
    cosine_sim = nn.CosineSimilarity(dim=1)
    similarities = cosine_sim(embedding_matrix, word_vec.expand(embedding_matrix.shape[0], -1))

    # 获取最相似的k个词的索引
    _, indices = torch.topk(similarities, k=min(top_k, len(similarities)))

    # 返回相应的词向量
    return embedding_matrix[indices]

# 注意力机制为相似词分配权重
def attention_weights(context_vector, similar_vectors):
    # 确保维度匹配
    if len(context_vector.shape) == 1:
        context_vector = context_vector.unsqueeze(0)

    # 添加投影层来处理维度不匹配
    if context_vector.size(-1) != similar_vectors.size(-1):
        projection = nn.Linear(context_vector.size(-1), similar_vectors.size(-1))
        context_vector = projection(context_vector)

    scores = torch.matmul(context_vector, similar_vectors.transpose(-2, -1))
    attention_weights = F.softmax(scores, dim=-1)
    return attention_weights

# 计算语义增强后的向量
def semantic_augmentation(context_vector, similar_vectors, attention_weights):
    augmented_vector = torch.matmul(attention_weights, similar_vectors)
    return augmented_vector

# 四、条件随机场层:进行序列标注

# 定义标签到ID的映射字典
tag2id = {
    'O': 0,           # 非实体
    'B-PERSON': 1,    # 人名的开始
    'I-PERSON': 2,    # 人名的中间部分
    'B-ORG': 3,       # 组织名的开始
    'I-ORG': 4,       # 组织名的中间部分
    'B-CARDINAL': 5,  # 数字的开始
    'I-CARDINAL': 6,  # 数字的中间部分
    'B-DATE': 7,      # 日期的开始
    'I-DATE': 8       # 日期的中间部分
}

# 反向映射ID到标签
id2tag = {v: k for k, v in tag2id.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_tags = len(tag2id)  # 标签数量
crf_layer = CRF(num_tags, batch_first=True).to(device)

# 损失函数
def compute_loss(emissions, tags, mask, crf_layer, device):
    tags = tags.to(device)
    mask = mask.to(device)
    emissions = emissions.to(device) 

    # 计算CRF层的loss
    loss = -crf_layer(emissions, tags, mask=mask, reduction='mean')
    return loss

# 预测标签
def predict_tags(emissions, mask, device):
    emissions = emissions.to(device)  
    mask = mask.to(device)  
    pred_tags = crf_layer.decode(emissions, mask=mask)
    return pred_tags

# 五、模型整合，将前面几个模块全部整合成一个模型
class NERModel(nn.Module):
    def __init__(self, zen_model, embedding_matrix, num_tags):
        super(NERModel, self).__init__()
        self.zen_model = zen_model
        self.embedding_matrix = embedding_matrix
        self.crf_layer = CRF(num_tags, batch_first=True)  # CRF层
        self.hidden_size = 768  # ZEN的隐藏层维度

        # 添加用于context_vectors的投影层，将1024映射到768
        self.context_projection = nn.Linear(1024, 768)

        # 添加用于embedding_matrix的投影层，确保embedding_matrix的维度与context_vectors一致
        self.projection = nn.Linear(embedding_matrix.shape[-1], self.hidden_size)

        # 添加门控层，确保维度匹配
        self.gate_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 添加输出层
        self.output_layer = nn.Linear(self.hidden_size, num_tags)

        # 添加语义增强层
        self.semantic_projection = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input_ids, attention_mask=None, device=None):
        self.embedding_matrix = self.embedding_matrix.to(device)

        # 1. 上下文编码
        outputs = self.zen_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        context_vectors = outputs.last_hidden_state

        # 将context_vectors投影到768维
        context_vectors = self.context_projection(context_vectors)

        # 2. 语义增强与相似词的获取
        batch_size, seq_len, _ = context_vectors.shape
        augmented_vectors = []

        # 确保embedding_matrix维度匹配
        working_embedding_matrix = self.projection(self.embedding_matrix)

        for i in range(batch_size):
            seq_augmented = []
            for j in range(seq_len):
                # 获取token的表示
                token_vector = context_vectors[i,j]

                # 获取相似词并计算注意力
                similar_words = get_similar_words(token_vector, working_embedding_matrix)  # [top_k, hidden_size]
                attention = attention_weights(token_vector.unsqueeze(0), similar_words)  # [1, top_k]
                augmented_vector = semantic_augmentation(token_vector, similar_words, attention)  # [hidden_size]
                # 确保增强向量维度匹配
                augmented_vector = self.semantic_projection(augmented_vector)
                seq_augmented.append(augmented_vector)

            # 将序列中的所有增强向量堆叠
            seq_tensor = torch.stack(seq_augmented)
            augmented_vectors.append(seq_tensor)

        # 将批次中的所有序列堆叠
        augmented_vectors = torch.stack(augmented_vectors)

        # 3. 门控机制结合
        # 确保维度匹配
        if len(augmented_vectors.shape) == 4:
            # 如果augmented_vectors是4维的，去掉多余的维度
            augmented_vectors = augmented_vectors.squeeze(2)

        # 拼接向量
        concat = torch.cat([context_vectors, augmented_vectors], dim=-1)

        # 计算门控值
        gate = torch.sigmoid(self.gate_layer(concat))

        # 应用门控
        gated_output = gate * context_vectors + (1 - gate) * augmented_vectors

        # 4. 映射到标签空间
        emissions = self.output_layer(gated_output)

        return emissions

# 六、评估函数，计算F1分数
def evaluate_model(model, data_loader, id2tag, device):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            emissions = model(input_ids, attention_mask, device=device)
            mask = attention_mask.type(torch.uint8).to(device)

            pred_tags = model.crf_layer.decode(emissions, mask=mask)

            for true, pred, m in zip(labels, pred_tags, mask):
                valid_len = m.sum().item()
                true = true[:valid_len].cpu().numpy()
                pred = np.array(pred[:valid_len])

                # 计算非O标签的比例
                non_o_ratio = sum(id2tag[t] != 'O' for t in true) / len(true)

                # 如果非O标签比例大于某个阈值，或随机保留一些全O的样本
                if non_o_ratio > 0 or np.random.random() < 0.2:  # 保留20%的全O样本
                    true_label = [id2tag[int(l)] for l in true]
                    pred_label = [id2tag[int(p)] for p in pred]
                    true_labels.extend(true_label)
                    pred_labels.extend(pred_label)

    if len(true_labels) == 0:
        return 0.0

    return f1_score(true_labels, pred_labels, average='weighted')

# 七、主函数：模型训练与评估
def main():
    # 1. 超参数设置
    train_file = '/content/drive/MyDrive/douyin_train.txt'
    val_file = '/content/drive/MyDrive/douyin_val.txt'
    test_file = '/content/drive/MyDrive/douyin_test.txt'
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    max_length = 128

    # 2. 加载ZEN模型和分词器
    tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-ZEN2-668M-Chinese')
    zen_model = BertModel.from_pretrained('IDEA-CCNL/Erlangshen-ZEN2-668M-Chinese')

    # 初始化embedding_matrix
    vocab_size = len(tokenizer.vocab)
    embedding_dim = 768  # 与ZEN模型的隐藏层大小相同
    embedding_matrix = torch.randn(vocab_size, embedding_dim)

    # 3. 加载数据集
    train_loader, val_loader, test_loader = load_datasets(train_file, val_file, test_file, tokenizer, tag2id, batch_size)

    # 4. 初始化模型
    model = NERModel(zen_model=BertModel.from_pretrained('IDEA-CCNL/Erlangshen-ZEN2-668M-Chinese'),
                     embedding_matrix=embedding_matrix,
                     num_tags=len(tag2id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5. 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 6. 训练模型
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            emissions = model(input_ids, attention_mask, device=device)
            mask = attention_mask.type(torch.uint8)
            loss = compute_loss(emissions, labels, mask, model.crf_layer, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

        # 验证集评估
        val_f1 = evaluate_model(model, val_loader, id2tag, device)
        print(f'Validation F1 Score after epoch {epoch+1}: {val_f1}')

    # 7. 测试模型
    test_f1 = evaluate_model(model, test_loader, id2tag, device)
    print(f'Test F1 Score: {test_f1}')

if __name__ == "__main__":
    main()