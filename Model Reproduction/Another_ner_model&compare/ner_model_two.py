import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchcrf import CRF
from seqeval.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast, BertModel
from torch.cuda.amp import autocast, GradScaler
from google.colab import drive
drive.mount('/content/drive')

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取BIOES格式的数据集
def load_data(file_path):
    sentences = []
    labels = []
    sentence = []
    label = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "":  # 空行表示句子结束
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                # 确保每行有两个元素：一个词和一个标签
                parts = line.strip().split()
                if len(parts) == 2:
                    word, tag = parts
                    sentence.append(word)
                    label.append(tag)

        if sentence:  # 处理文件末尾没有空行的情况
            sentences.append(sentence)
            labels.append(label)

    return sentences, labels


# 加载训练、验证和测试数据
train_sentences, train_labels = load_data('/content/drive/MyDrive/douyin_train.txt')
val_sentences, val_labels = load_data('/content/drive/MyDrive/douyin_val.txt')
test_sentences, test_labels = load_data('/content/drive/MyDrive/douyin_test.txt')

# 使用 Fast BERT 分词器
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)

def tokenize_and_embed(sentences):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, is_split_into_words=True)
    inputs = {key: value.to(device) for key, value in inputs.items()} 

    with torch.no_grad():
        outputs = bert_model(**inputs)

    return outputs.last_hidden_state

# 检查数据集中所有出现的标签
def check_all_labels(data):
    all_labels = set()
    for labels in data:
        all_labels.update(labels)
    return all_labels

# 检查训练、验证和测试集中的所有标签，以便于后续的标签映射字典的写入
train_labels_set = check_all_labels(train_labels)
val_labels_set = check_all_labels(val_labels)
test_labels_set = check_all_labels(test_labels)

print("训练集中的标签：", train_labels_set)
print("验证集中的标签：", val_labels_set)
print("测试集中的标签：", test_labels_set)

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tag2id):
        self.sentences = sentences
        self.labels = labels
        self.tag2id = tag2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = [self.tag2id[tag] for tag in self.labels[idx]]  # 标签映射为ID
        return {"sentences": sentence, "labels": label}

# 对齐标签与 BERT 分词后的子词。
def align_labels_with_subwords(labels, word_ids):
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # 对于特殊符号（如 [CLS], [SEP] 等），我们忽略
            aligned_labels.append(-100)  # -100 是忽略的标签
        elif word_id != previous_word_id:
            # 对齐第一个子词的标签
            aligned_labels.append(labels[word_id])
        else:
            # 对于同一个单词的其他子词，重复标签
            aligned_labels.append(labels[word_id])

        previous_word_id = word_id

    return aligned_labels

def collate_fn(batch):
    sentences = [item['sentences'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 使用 BERT 分词器对句子进行填充
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt',
        is_split_into_words=True,
        max_length=512  # 添加最大长度限制
    )

    # 获取每个句子的词到token的映射
    batch_size = len(sentences)
    max_length = encoded['input_ids'].size(1)

    # 创建与token序列等长的标签序列
    padded_labels = torch.full((batch_size, max_length), tag2id['O'], device=device)

    # 对每个样本处理标签
    for i, (label_seq, input_seq) in enumerate(zip(labels, encoded['input_ids'])):
        # 将标签序列截断或填充到与input_ids相同的长度
        seq_length = min(len(label_seq), max_length)
        padded_labels[i, :seq_length] = torch.tensor(label_seq[:seq_length], device=device)

    # 将输入移到设备
    inputs = {
        'input_ids': encoded['input_ids'].to(device),
        'attention_mask': encoded['attention_mask'].to(device)
    }

    return inputs, padded_labels

# 标签映射字典
train_tags = {'B-QUANTITY', 'E-FAC', 'S-LANGUAGE', 'I-MONEY', 'E-NORP', 'I-PRODUCT', 'E-EVENT', 'B-TIME', 'E-LOC', 'E-GPE', 'I-ORDINAL', 'E-PERCENT', 'E-CARDINAL', 'E-LANGUAGE', 'S-GPE', 'E-DATE', 'B-LANGUAGE', 'O', 'I-QUANTITY', 'E-ORG', 'E-PRODUCT', 'B-GPE', 'E-TIME', 'B-WORK_OF_ART', 'B-PERSON', 'B-PERCENT', 'I-TIME', 'B-NORP', 'B-LOC', 'E-QUANTITY', 'B-DATE', 'B-PRODUCT', 'I-GPE', 'I-DATE', 'E-MONEY', 'B-MONEY', 'I-CARDINAL', 'S-ORG', 'I-WORK_OF_ART', 'B-CARDINAL', 'I-PERCENT', 'B-EVENT', 'E-ORDINAL', 'I-PERSON', 'I-NORP', 'E-WORK_OF_ART', 'I-LOC', 'S-TIME', 'I-LANGUAGE', 'B-FAC', 'E-PERSON', 'S-CARDINAL', 'B-ORG', 'I-FAC', 'B-ORDINAL', 'I-ORG', 'I-EVENT', 'S-PERSON'}
val_tags = {'B-QUANTITY', 'E-FAC', 'S-LANGUAGE', 'I-MONEY', 'E-NORP', 'I-PRODUCT', 'E-EVENT', 'B-TIME', 'E-LOC', 'E-GPE', 'E-PERCENT', 'S-GPE', 'E-CARDINAL', 'E-LANGUAGE', 'I-ORDINAL', 'E-DATE', 'B-LANGUAGE', 'O', 'I-QUANTITY', 'E-ORG', 'E-PRODUCT', 'B-GPE', 'E-TIME', 'B-WORK_OF_ART', 'B-PERSON', 'B-NORP', 'B-PERCENT', 'I-TIME', 'B-LOC', 'E-QUANTITY', 'B-DATE', 'B-PRODUCT', 'I-GPE', 'I-DATE', 'E-MONEY', 'B-MONEY', 'I-CARDINAL', 'I-WORK_OF_ART', 'B-CARDINAL', 'B-EVENT', 'E-ORDINAL', 'I-PERSON', 'E-WORK_OF_ART', 'I-NORP', 'I-LOC', 'I-LANGUAGE', 'B-FAC', 'E-PERSON', 'B-ORG', 'S-CARDINAL', 'I-FAC', 'B-ORDINAL', 'I-ORG', 'I-EVENT', 'S-PERSON'}
test_tags = {'B-QUANTITY', 'E-FAC', 'I-MONEY', 'E-NORP', 'I-PRODUCT', 'E-EVENT', 'B-TIME', 'E-LOC', 'E-GPE', 'E-PERCENT', 'S-GPE', 'I-QUANTITY', 'E-LANGUAGE', 'I-ORDINAL', 'E-DATE', 'B-LANGUAGE', 'O', 'E-ORG', 'E-PRODUCT', 'B-GPE', 'E-TIME', 'B-WORK_OF_ART', 'B-PERSON', 'B-NORP', 'B-PERCENT', 'I-TIME', 'B-LOC', 'E-QUANTITY', 'B-DATE', 'B-PRODUCT', 'I-GPE', 'I-DATE', 'I-ORG', 'B-MONEY', 'E-MONEY', 'I-CARDINAL', 'I-WORK_OF_ART', 'B-CARDINAL', 'I-PERCENT', 'B-EVENT', 'E-ORDINAL', 'S-DATE', 'I-PERSON', 'E-WORK_OF_ART', 'I-NORP', 'I-LOC', 'B-FAC', 'E-PERSON', 'S-CARDINAL', 'B-ORG', 'I-FAC', 'B-ORDINAL', 'E-CARDINAL', 'I-EVENT', 'S-PERSON'}

# 取三者并集
all_tags = train_tags.union(val_tags).union(test_tags)

# 为标签分配ID
tag2id = {tag: i for i, tag in enumerate(sorted(all_tags))}

# 创建数据集时传入 tag2id
train_dataset = NERDataset(train_sentences, train_labels, tag2id)
val_dataset = NERDataset(val_sentences, val_labels, tag2id)
test_dataset = NERDataset(test_sentences, test_labels, tag2id)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# 共享-私有特征提取器
class SharedPrivateBiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_labels):
        super(SharedPrivateBiLSTM, self).__init__()
        self.shared_bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.private_bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, embeddings):
        shared_output, _ = self.shared_bilstm(embeddings)
        private_output, _ = self.private_bilstm(embeddings)
        return shared_output, private_output

# 自注意力机制:多头注意力机制
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim * 2, num_heads)

    def forward(self, lstm_output):
        # lstm_output should have shape (batch_size, sequence_length, hidden_dim * 2)
        attn_output, _ = self.multihead_attn(lstm_output, lstm_output, lstm_output)
        return attn_output

# CRF层
class CRFLayer(nn.Module):
    def __init__(self, num_labels):
        super(CRFLayer, self).__init__()
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, logits, tags):
        return -self.crf(logits, tags)

    def decode(self, logits):
        return self.crf.decode(logits)

# 任务判别器
class TaskDiscriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(TaskDiscriminator, self).__init__()
        self.fc = nn.Linear(hidden_dim * 2, 2)  # 2表示两个任务
        self.softmax = nn.Softmax(dim=1)

    def forward(self, shared_features):
        # shared_features shape: (batch_size, sequence_length, hidden_dim * 2)
        # 使用平均池化，得到每个序列的表示
        pooled_features = torch.mean(shared_features, dim=1)  # (batch_size, hidden_dim * 2)
        logits = self.fc(pooled_features)
        return self.softmax(logits)

# 整合模型
class NERModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_labels, num_heads):
        super(NERModel, self).__init__()

        # 共享BiLSTM
        self.shared_private_bilstm = nn.LSTM(
            embedding_dim, hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # MultiheadAttention 需要 `hidden_​​dim * 2` 作为输入和输出维度
        self.self_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads, batch_first=True)

        # 特定任务层
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

        # 任务判别器
        self.task_discriminator = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=-1)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        # 共享特征
        shared_output, _ = self.shared_private_bilstm(x)
        attn_output, _ = self.self_attention(shared_output, shared_output, shared_output)
        # 特定任务的预测
        logits = self.classifier(attn_output)

        if labels is not None:
            # 训练模式：返回损失
            loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return loss
        else:
            # 预测模式：返回logits
            return logits

# 直接使用已经分词的输入获取BERT嵌入
def get_bert_embeddings(input_ids, attention_mask):
    
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state

# 评估函数
def evaluate(model, dataloader, mode='val'):
    model.eval()
    all_preds = []
    all_labels = []

    id2tag = {id: tag for tag, id in tag2id.items()}

    with torch.no_grad():
        with autocast():
            for inputs, labels in dataloader:
                embeddings = get_bert_embeddings(inputs['input_ids'], inputs['attention_mask'])
                mask = inputs['attention_mask']
                logits = model(embeddings)  # 确保模型返回logits

                # 处理每个序列
                for logit, label, m in zip(logits, labels, mask):
                    valid_length = m.sum().item()
                    # 获取预测的类别
                    pred = torch.argmax(logit[:valid_length], dim=-1)

                    # 转换为标签
                    pred_tags = [id2tag[p.item()] for p in pred]
                    label_tags = [id2tag[l.item()] for l in label[:valid_length]]

                    all_preds.append(pred_tags)
                    all_labels.append(label_tags)

    return f1_score(all_labels, all_preds)

# 模型训练与评估
def train_model(model, train_dataloader, val_dataloader, test_dataloader, optimizer, num_epochs):
    scaler = GradScaler()
    best_val_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            with autocast():
                embeddings = get_bert_embeddings(inputs['input_ids'], inputs['attention_mask'])
                task_loss = model(embeddings, labels)

                # BiLSTM 的共享输出
                shared_output, _ = model.shared_private_bilstm(embeddings)
                shared_output, _ = model.self_attention(shared_output, shared_output, shared_output)

                # 池化以减少序列长度维度
                pooled_output = torch.mean(shared_output, dim=1)  
                # 任务判别器输出
                discriminator_output = model.task_discriminator(pooled_output)  

                # 建立统一目标
                batch_size = discriminator_output.size(0)
                uniform_targets = torch.ones(batch_size, 2, device=device) / 2

                # 计算对抗损失
                adv_loss = torch.mean(torch.sum(-uniform_targets * torch.log(discriminator_output + 1e-6), dim=1))

                # 总损失=任务损失*+对抗损失*0.1
                loss = task_loss + 0.1 * adv_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")

        val_f1 = evaluate(model, val_dataloader, mode='val')
        print(f"Validation F1 Score after epoch {epoch + 1}: {val_f1}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    test_f1 = evaluate(model, test_dataloader, mode='test')
    print(f"Test F1 Score: {test_f1}")

    return best_val_f1, test_f1

# 初始化模型和优化器
model = NERModel(
    embedding_dim=768,
    hidden_dim=120,
    num_labels=len(tag2id),
    num_heads=8
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 开始训练
best_val_f1, test_f1 = train_model(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    optimizer,
    num_epochs=50
)