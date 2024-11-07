import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 클래스 정의
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.drop('price', axis=1).values
        self.target = self.data['price'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.target[idx]])

# 트랜스포머 기반 회귀 모델 정의
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

# 모델 학습 함수
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 모델 평가 함수
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# 어텐션 맵 추출 및 시각화 함수
def visualize_attention(model, sample_input, feature_names):
    model.eval()
    with torch.no_grad():
        # 임베딩 레이어 통과
        embedded = model.embedding(sample_input).unsqueeze(1)
        
        # 트랜스포머 레이어의 첫 번째 레이어에서 어텐션 맵 추출
        attention_map = model.transformer.layers[0].self_attn(embedded, embedded, embedded)[1]
        attention_map = attention_map.squeeze(0).cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_map, xticklabels=feature_names, yticklabels=feature_names, cmap='YlGnBu')
    plt.title('Attention Map')
    plt.show()

# 메인 실행 코드
def main():
    # 데이터 로드 및 전처리
    train_dataset = CSVDataset('train.csv')
    test_dataset = CSVDataset('test.csv')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 모델 초기화
    input_dim = train_dataset.features.shape[1]
    model = TransformerRegressor(input_dim=input_dim, hidden_dim=64, num_layers=3, num_heads=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # 어텐션 맵 시각화
    sample_input = torch.FloatTensor(train_dataset[0][0]).unsqueeze(0).to(device)
    feature_names = train_dataset.data.drop('price', axis=1).columns
    visualize_attention(model, sample_input, feature_names)

if __name__ == '__main__':
    main()