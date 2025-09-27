import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, ErnieModel
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

    def __len__(self):
        return len(self.sentences)

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

# 二、使用ERNIE模型
# 初始化ERNIE模型和分词器
from huggingface_hub import login
login(token="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")
tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-medium-zh")
ernie_model = ErnieModel.from_pretrained("nghuyong/ernie-3.0-medium-zh")

# 三、使用多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention_heads = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads) for _ in range(num_heads)])

    def forward(self, context_vectors):
        attentions = []
        for head in self.attention_heads:
            attn_output, _ = head(context_vectors, context_vectors, context_vectors)
            attentions.append(attn_output)
        return torch.cat(attentions, dim=-1)


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
    # 确保tags和mask在同一设备上
    tags = tags.to(device)
    mask = mask.to(device)
    emissions = emissions.to(device)  # 将emissions也移动到device

    # 计算CRF层的loss
    loss = -crf_layer(emissions, tags, mask=mask, reduction='mean')
    return loss

# 五、定义NER模型（使用ERNIE和多头注意力机制）
class NERModel(nn.Module):
    def __init__(self, ernie_model, embedding_matrix, num_tags, num_heads=8, dropout_rate=0.1):
        super(NERModel, self).__init__()
        self.ernie_model = ernie_model
        self.embedding_matrix = embedding_matrix
        self.crf_layer = CRF(num_tags, batch_first=True)
        self.hidden_size = 768  # ERNIE的隐藏层维度
        self.num_heads = num_heads

        # 添加多头注意力机制
        self.multihead_attention = MultiHeadAttention(self.hidden_size, num_heads=self.num_heads)

        # 添加投影层，将多头注意力输出的维度映射回隐藏层维度
        self.context_projection = nn.Linear(self.hidden_size * num_heads, self.hidden_size)

        # 添加Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 输出层
        self.output_layer = nn.Linear(self.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask=None, device=None):
        self.embedding_matrix = self.embedding_matrix.to(device)

        # 上下文编码（使用ERNIE）
        outputs = self.ernie_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        context_vectors = outputs.last_hidden_state

        # 多头注意力机制
        attention_output = self.multihead_attention(context_vectors)

        # 投影到原始维度
        context_vectors = self.context_projection(attention_output)

        # Dropout
        context_vectors = self.dropout(context_vectors)

        # 映射到标签空间
        emissions = self.output_layer(context_vectors)

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

# 使用混合精度训练
scaler = torch.cuda.amp()

# 七、主函数：模型训练与评估
def main():
    # 1. 超参数设置
    train_file = '/content/drive/MyDrive/douyin_train.txt'
    val_file = '/content/drive/MyDrive/douyin_val.txt'
    test_file = '/content/drive/MyDrive/douyin_test.txt'
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.0001
    max_length = 128

    # 2. 加载模型和分词器

    tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-medium-zh")
    ernie_model = ErnieModel.from_pretrained("nghuyong/ernie-3.0-medium-zh")

    # 初始化embedding_matrix
    vocab_size = len(tokenizer.vocab)
    embedding_dim = 768  # 与ERNIE模型的隐藏层大小相同
    embedding_matrix = torch.randn(vocab_size, embedding_dim)

    # 3. 加载数据集
    train_loader, val_loader, test_loader = load_datasets(train_file, val_file, test_file, tokenizer, tag2id, batch_size)

    # 4. 初始化模型
    model = NERModel(ernie_model=ErnieModel.from_pretrained("nghuyong/ernie-3.0-medium-zh"),
                     embedding_matrix=embedding_matrix,
                     num_tags=len(tag2id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5. 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

   # 训练过程
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            # 混合精度训练
            with torch.amp.autocast(device_type='cuda'):
                emissions = model(input_ids, attention_mask, device=device)
                mask = attention_mask.type(torch.uint8)
                loss = compute_loss(emissions, labels, mask, model.crf_layer, device)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

        # 验证集评估
        val_f1 = evaluate_model(model, val_loader, id2tag, device)
        print(f'Validation F1 Score after epoch {epoch+1}: {val_f1}')

    # 测试模型
    test_f1 = evaluate_model(model, test_loader, id2tag, device)
    print(f'Test F1 Score: {test_f1}')

if __name__ == "__main__":
    main()