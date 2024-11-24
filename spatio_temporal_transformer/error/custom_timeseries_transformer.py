import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import seaborn as sns

class CustomTimeSeriesDataset(Dataset):
    def __init__(self, data, num_timesteps, target_window, target_node, target_feature):
        self.num_timesteps = num_timesteps
        self.target_window = target_window
        self.target_node = target_node
        self.target_feature = target_feature
        
        self.X, self.y = self._prepare_data(data)
    
    def _prepare_data(self, data):
        X = []
        y = []
        target_col = f"{self.target_node}_{self.target_feature}"
        
        for i in range(len(data) - self.num_timesteps - self.target_window + 1):
            X.append(data[i:i+self.num_timesteps].values.flatten())
            y.append(data[i+self.num_timesteps:i+self.num_timesteps+self.target_window][target_col].values)
        
        # return np.array(X), np.array(y)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # return self.X[index], self.y[index]
        return torch.from_numpy(self.X[index]), torch.from_numpy(self.y[index])
    
    def get_num_features(self):
        return self.X.shape[1] // self.num_timesteps

def load_and_prepare_data(file_path, num_timesteps, target_window, target_node, target_feature, test_size=0.2):
    # CSV 파일 로드
    df = pd.read_csv(file_path, index_col=0)
    path_mergeddf = './preprocessed_data_slotted/89/'

    # input, target feature로 사용할 column 선택
    selected_features = ["Cpu", "Memory", "Throughput", "ResponseTime"]
    df = df[[col for col in df.columns if any(col.endswith('_' + feature) for feature in selected_features)]]

    # csv 파일로 저장
    df.to_csv(path_mergeddf+'selected_merged_df.csv', date_format='%Y-%m-%d %H:%M:%S', index=False)

    # 학습 및 테스트 세트 분할
    train_size = int((1 - test_size) * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # 데이터셋 생성
    train_dataset = CustomTimeSeriesDataset(train_df, num_timesteps, target_window, target_node, target_feature)
    test_dataset = CustomTimeSeriesDataset(test_df, num_timesteps, target_window, target_node, target_feature)
    
    # 데이터 정규화
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    train_dataset.X = scaler_X.fit_transform(train_dataset.X)
    test_dataset.X = scaler_X.transform(test_dataset.X)
    
    train_dataset.y = scaler_y.fit_transform(train_dataset.y.reshape(-1, 1)).reshape(train_dataset.y.shape)
    test_dataset.y = scaler_y.transform(test_dataset.y.reshape(-1, 1)).reshape(test_dataset.y.shape)

    return train_dataset, test_dataset, scaler_X, scaler_y


def create_dataloaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def save_datasets(train_dataset, test_dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.save_to_csv(os.path.join(output_dir, "train_dataset.csv"))
    test_dataset.save_to_csv(os.path.join(output_dir, "test_dataset.csv"))


# 모델 정의
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers
        )
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src).unsqueeze(0)  # Add sequence length dimension
        output = self.transformer(src)
        return self.output_layer(output.squeeze(0))
    

# 학습 함수
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 평가 함수
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
    return total_loss / len(test_loader)


# 메인 함수
def main():
    file_path = "./preprocessed_data_slotted/89/merged_df.csv"
    # 하이퍼파라미터
    num_timesteps = 3  # t1, t2, t3를 입력으로 사용
    target_window = 5  # t4부터 t33까지 예측
    target_node = "B2"  # B2 노드 예측
    target_feature = "Throughput"
    
    batch_size = 32
    num_epochs = 10
    lr = 0.001
    d_model = 128
    nhead = 4
    num_layers = 3
    dim_feedforward = 512

    # 데이터 로드 및 전처리
    train_dataset, test_dataset, scaler_X, scaler_y = load_and_prepare_data(
        file_path, num_timesteps, target_window, target_node, target_feature
    )
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size)

     # 모델 초기화
    input_dim = train_dataset.get_num_features() * num_timesteps
    model = TimeSeriesTransformer(input_dim=input_dim, output_dim=target_window, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)
    
    # 모델을 float32로 변환
    model = model.float()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    print("Training completed.")

    # 추론
    model.eval()
    predictions = []
    actual_values = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            predictions.append(output.cpu().numpy())
            actual_values.append(batch_y.numpy())

    # numpy 배열로 변환
    predictions = np.concatenate(predictions, axis=0)
    actual_values = np.concatenate(actual_values, axis=0)

    # 예측값과 실제값 역변환
    predictions_unscaled = scaler_y.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    actual_values_unscaled = scaler_y.inverse_transform(actual_values.reshape(-1, 1)).reshape(actual_values.shape)

    # 결과 출력
    print("Predictions shape:", predictions_unscaled.shape)
    print("Actual values shape:", actual_values_unscaled.shape)
    print("Sample predictions:", predictions_unscaled[:5].flatten())
    print("Sample actual values:", actual_values_unscaled[:5].flatten())

    # 평가 메트릭 계산 (예: MSE, MAE)
    mse = np.mean((predictions_unscaled - actual_values_unscaled)**2)
    mae = np.mean(np.abs(predictions_unscaled - actual_values_unscaled))
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

if __name__ == "__main__":
    main()

    # # 데이터 로드 및 전처리
    # train_dataset, test_dataset, scaler = load_and_prepare_data(
    #     file_path, num_timesteps, target_window, target_node, target_feature
    # )
    # train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size)

    # # 데이터 형태 확인
    # for batch_x, batch_y in train_loader:
    #     print("Input shape:", batch_x.shape)
    #     print("Target shape:", batch_y.shape)
    #     break
    # print("Number of training samples:", len(train_dataset))
    # print("Number of test samples:", len(test_dataset))

    # # 모델 초기화
    # input_dim = train_dataset.get_num_features() * num_timesteps
    # model = TimeSeriesTransformer(input_dim=input_dim, output_dim=target_window, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    # # 학습
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # for epoch in range(num_epochs):
    #     train_loss = train(model, train_loader, criterion, optimizer, device)
    #     test_loss = evaluate(model, test_loader, criterion, device)
    #     print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # print("Training completed.")

    # # 추론
    # model.eval()
    # with torch.no_grad():
    #     for batch_x, batch_y in test_loader:
    #         batch_x = batch_x.to(device)
    #         output = model(batch_x)
    #         output = output.cpu().numpy()
    #         # output = scaler.inverse_transform(output)
    #         print(output)