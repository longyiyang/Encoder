import jieba
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.translate.bleu_score import sentence_bleu
from dataset import TextDataset
from torch.utils.data import Dataset, DataLoader

# 读取数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if len(line.strip()) >= 4]  # 只保留字数大于等于4的行

# 分词
def tokenize(texts):
    return [' '.join(jieba.cut(text)) for text in texts]

# 创建词汇表
def create_vocab(texts):
    vocab = set()
    for text in texts:
        words = text.split()
        vocab.update(words)
    vocab = sorted(vocab)  # 按字典顺序排序
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word, vocab_size

# 数据预处理：加载数据，分词，索引化
def preprocess_data(file_path):
    texts = load_data(file_path)
    tokenized_texts = tokenize(texts)
    word_to_idx, idx_to_word, vocab_size = create_vocab(tokenized_texts)
    
    # 转换为索引列表
    indexed_texts = [[word_to_idx[word] for word in text.split()] for text in tokenized_texts]
    
    return indexed_texts, word_to_idx, idx_to_word, vocab_size

# 加载并预处理数据
train_data, word_to_idx, idx_to_word, vocab_size = preprocess_data('./data/data.txt')

# 划分训练集和测试集
train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

class TransformerAutoEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_heads=8, num_layers=6, target_dim=6):
        super(TransformerAutoEncoder, self).__init__()

        # 定义嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 定义 Transformer 编码器
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # 定义映射层将编码器输出压缩为 target_dim
        self.fc1 = nn.Linear(embed_size, target_dim)

        # 定义解码器
        self.fc2 = nn.Linear(target_dim, embed_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc3 = nn.Linear(embed_size, vocab_size)
        
    def forward(self, src, tgt):
        # 编码器部分
        src = self.embedding(src)
        encoded = self.transformer_encoder(src)

        # 压缩：映射到 target_dim
        compressed = self.fc1(encoded.mean(dim=0))  # 平均池化后的压缩表示

        # 解码器部分
        tgt = self.fc2(compressed).unsqueeze(0).repeat(encoded.size(0), 1, 1)  # 转换为和原始输入相同的形状
        decoded = self.transformer_decoder(tgt, encoded)
        
        # 输出层：预测每个位置的词
        output = self.fc3(decoded)
        return output

# 设置模型
model = TransformerAutoEncoder(vocab_size=vocab_size, embed_size=256, target_dim=6)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将数据转换为Tensor格式
def prepare_data_for_training(data, seq_len=20):
    # Padding或截断到固定长度
    padded_data = [d[:seq_len] if len(d) >= seq_len else d + [0] * (seq_len - len(d)) for d in data]
    return torch.tensor(padded_data)

# 创建训练集和测试集的 Dataset
train_dataset = TextDataset(train_data, seq_len=20)
test_dataset = TextDataset(test_data, seq_len=20)


# 判断是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型转移到设备上
model = model.to(device)

# 准备数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


import torch

# 假设有一个全局的device变量和criterion、optimizer等已经定义

def train(model, train_loader, test_loader, epochs=10, save_path="best_model.pth"):
    model.train()  # 设置模型为训练模式
    best_loss = float('inf')  # 初始化最好的测试损失为无穷大
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            # 获取输入数据和标签
            inputs = batch.to(device)
            targets = batch.to(device)  # 目标和输入相同
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播：通过模型进行预测
            output = model(inputs, inputs)
            
            # 计算损失
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 累积损失
            epoch_loss += loss.item()
        
        # 打印每个 epoch 的平均损失
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")
        
        # 每隔一定周期测试模型在测试集上的表现（可选）
        if epoch % 5 == 0:
            test_loss = evaluate(model, test_loader)
            
            # 保存最佳模型（如果当前模型的测试损失更小）
            if test_loss < best_loss:
                best_loss = test_loss
                print(f"Saving model with improved test loss: {test_loss}")
                torch.save(model.state_dict(), save_path)

# 测试模型的性能
def evaluate(model, test_loader):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    with torch.no_grad():  # 禁用梯度计算
        for batch in test_loader:
            inputs = batch.to(device)
            targets = batch.to(device)
            
            # 前向传播：通过模型进行预测
            output = model(inputs, inputs)
            
            # 计算损失
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}")
    return avg_loss

# 开始训练
train(model, train_loader, test_loader, epochs=10)

# 加载模型的函数
def load_model(model, save_path="best_model.pth"):
    model.load_state_dict(torch.load(save_path))
    model.eval()  # 切换为评估模式
    print(f"Model loaded from {save_path}")
    return model

# 压缩并解码
def compress_and_decode(model, input_text, word_to_idx, idx_to_word):
    # Convert words to indices
    input_tensor = torch.tensor([word_to_idx.get(word, 0) for word in input_text]).unsqueeze(0)  # Shape: (1, seq_len)
    
    # Move the tensor to the same device as the model
    input_tensor = input_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        # First, we need to embed the input tensor to get (seq_len, batch_size, embed_size)
        input_tensor = model.embedding(input_tensor)  # Embedding: Shape becomes (1, seq_len, embed_size)

        # Now, the input_tensor has the correct shape for the transformer_encoder
        encoded = model.transformer_encoder(input_tensor.permute(1, 0, 2))  # (seq_len, batch_size, embed_size)

        # Compress the encoded output to target_dim
        compressed = model.fc1(encoded.mean(dim=0))  # Mean pooling, then reduce to target_dim
        
        # Decode the compressed representation
        decoded = model.fc2(compressed).unsqueeze(0).repeat(encoded.size(0), 1, 1)  # Repeat to match input shape
        decoded_output = model.fc3(decoded)
        
        # Get the most probable word indices
        decoded_indices = decoded_output.argmax(dim=-1).squeeze(0)
    
    # Convert indices back to words
    decoded_words = [idx_to_word[idx.item()] for idx in decoded_indices]
    return ' '.join(decoded_words)


# 计算BLEU分数
def calculate_bleu_score(original, decoded):
    original_tokens = original
    decoded_tokens = decoded
    return sentence_bleu([original_tokens], decoded_tokens)

# Fixing the sample_text to use the actual tokens (words) from the test data, not indices
sample_text = test_data[2]  # This is a list of indices, so convert it to words

# Convert the indices back to words
sample_text_words = [idx_to_word[idx] for idx in sample_text]

compressed_result = compress_and_decode(model, sample_text_words, word_to_idx, idx_to_word)

print("原始文本：", ' '.join(sample_text_words))  # Joining words with spaces for printing
print("压缩并解码后的文本：", compressed_result)

# Calculate BLEU score
bleu_score = calculate_bleu_score(sample_text_words, compressed_result.split())
print(f"BLEU Score: {bleu_score}")

