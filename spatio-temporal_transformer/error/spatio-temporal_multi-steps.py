import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import List, Tuple, Dict
import json
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceDataset(Dataset):
    def __init__(self, data, input_timesteps, forecast_horizon, target_node, target_feature, selected_features=None):
        """
        Args:
            data: DataFrame with columns like 'B1_Memory', 'B2_ServiceTime'
            input_timesteps: number of past timesteps to use as input (e.g., t, t+1, t+2)
            forecast_horizon: number of future timesteps to predict (e.g., t+3, t+4 for horizon=2)
            target_node: target node name (e.g., 'B1')
            target_feature: target feature name (e.g., 'ResponseTime')
            selected_features: list of features to include (e.g., ['Cpu', 'Memory'])
        """
        self.input_timesteps = input_timesteps
        self.forecast_horizon = forecast_horizon
        self.target_node = target_node
        self.target_feature = target_feature
        self.selected_features = selected_features or ["Cpu", "Memory", "Throughput", "ResponseTime"]
        
        # 노드 리스트 추출
        self.node_list = self._get_unique_nodes(data)
        
        # 선택된 컬럼만 필터링
        self.filtered_data = self._filter_columns(data)
        
        # 데이터 준비
        self.X, self.y = self._prepare_data(self.filtered_data)
    
    def _get_unique_nodes(self, data):
        """데이터에서 unique한 노드 이름 추출"""
        nodes = set()
        for col in data.columns:
            node = col.split('_')[0]  # B1_Memory -> B1
            nodes.add(node)
        return sorted(list(nodes))
    
    def _filter_columns(self, data):
        """선택된 feature들만 포함하는 데이터프레임 반환"""
        selected_cols = [
            col for col in data.columns 
            if any(col.endswith('_' + feature) for feature in self.selected_features)
        ]
        return data[selected_cols]
    
    def _prepare_data(self, data):
        X = []
        y = []
        target_col = f"{self.target_node}_{self.target_feature}"
        
        # 입력과 예측 구간의 총 길이
        total_window = self.input_timesteps + self.forecast_horizon
        
        for i in range(len(data) - total_window + 1):
            # 입력 데이터 준비 (t ~ t+2)
            input_window = data[i:i+self.input_timesteps].values
            X.append(input_window.flatten())
            
            # 타겟 데이터 준비 (t+3 ~ t+4)
            target_values = data[i+self.input_timesteps:i+total_window][target_col].values
            y.append(target_values)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(self.y[index])
    
    @property
    def num_nodes(self):
        return len(self.node_list)
    
    @property
    def num_features(self):
        return len(self.selected_features)
    
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, 
                 num_nodes: int,
                 num_features: int,
                 input_timesteps: int,
                 forecast_horizon: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.input_timesteps = input_timesteps
        self.forecast_horizon = forecast_horizon
        
        # 임베딩 레이어
        self.feature_embedding = nn.Linear(1, d_model)
        
        # 각 인코딩
        self.temporal_encoding = nn.Parameter(torch.zeros(1, input_timesteps, d_model))
        self.node_encoding = nn.Parameter(torch.zeros(1, num_nodes, d_model))
        self.feature_encoding = nn.Parameter(torch.zeros(1, num_features, d_model))
        
        # Transformer 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 출력 레이어
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, forecast_horizon)
        )
        
        # attention weights 저장용
        self.attention_weights = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 입력 reshape: (batch, timesteps * nodes * features) -> (batch, timesteps, nodes, features)
        x = x.view(batch_size, self.input_timesteps, self.num_nodes, self.num_features)
        
        # feature 임베딩을 위해 차원 추가
        x = x.unsqueeze(-1)  # (batch, timesteps, nodes, features, 1)
        x = self.feature_embedding(x)  # (batch, timesteps, nodes, features, d_model)
        
        # 인코딩 추가
        x = x + self.temporal_encoding.unsqueeze(2).unsqueeze(3)
        x = x + self.node_encoding.unsqueeze(1).unsqueeze(3)
        x = x + self.feature_encoding.unsqueeze(1).unsqueeze(2)
        
        # Transformer 입력을 위한 reshape
        x = x.view(batch_size, self.input_timesteps * self.num_nodes * self.num_features, -1)
        
        # Transformer 통과
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # 회귀 예측
        return self.regression_head(x)
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """어텐션 가중치 추출"""
        self.attention_weights = []
        
        def hook_fn(module, input, output):
            self.attention_weights.append(output[1].detach())
        
        hooks = []
        for layer in self.transformer_encoder.layers:
            hook = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            self.forward(x)
        
        # Hooks 제거
        for hook in hooks:
            hook.remove()
            
        return self.attention_weights
    
class ExperimentManager:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 save_dir: str = "results"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        for batch_data, batch_targets in self.train_loader:
            batch_data = batch_data.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in self.val_loader:
                batch_data = batch_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_targets)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int, patience: int = 10):
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"best_model.pth")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= patience:
                logger.info("Early stopping triggered")
                break
                
        self.plot_training_history()
    
    def test(self) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data, batch_targets in self.test_loader:
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_targets.numpy())
                
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        return calculate_metrics(all_targets, all_preds)
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig(self.save_dir / 'training_history.png')
        plt.close()
        
    def visualize_attention(self, 
                          sample_data: torch.Tensor,
                          node_names: List[str],
                          feature_names: List[str],
                          layer_idx: int = 0):
        """특정 레이어의 어텐션 맵 시각화"""
        self.model.eval()
        attention_weights = self.model.get_attention_weights(sample_data)
        
        # 레이어의 어텐션 가중치 선택
        layer_attention = attention_weights[layer_idx][0]  # first batch
        
        # 데이터 구조에 맞게 reshape
        num_timesteps = self.model.input_timesteps
        num_nodes = self.model.num_nodes
        num_features = self.model.num_features
        
        # 전체 레이블 생성
        labels = []
        for t in range(num_timesteps):
            for n in node_names:
                for f in feature_names:
                    labels.append(f"t{t+1}_{n}_{f}")
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(layer_attention.cpu().numpy(),
                   xticklabels=labels,
                   yticklabels=labels,
                   cmap='viridis')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title(f'Attention Weights (Layer {layer_idx+1})')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'attention_map_layer_{layer_idx+1}.png')
        plt.close()


def calculate_metrics(y_true, y_pred):
    """
    시퀀스 예측에 대한 평가 메트릭 계산
    
    Args:
        y_true: shape (N, forecast_horizon)
        y_pred: shape (N, forecast_horizon)
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        # 각 시점별 메트릭
        'step_rmse': [
            np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            for i in range(y_true.shape[1])
        ]
    }
    return metrics

def prepare_data(file_path, input_timesteps, forecast_horizon, target_node, target_feature, 
                selected_features=None, test_size=0.2, val_size=0.1):
    """
    데이터 준비 및 전처리
    
    Args:
        file_path: 데이터 파일 경로
        input_timesteps: 입력으로 사용할 과거 시점 수
        forecast_horizon: 예측할 미래 시점 수
        target_node: 예측 대상 노드 (e.g., 'B1')
        target_feature: 예측 대상 성능 지표 (e.g., 'ResponseTime')
        selected_features: 사용할 성능 지표 리스트
        test_size: 테스트 세트 비율
        val_size: 검증 세트 비율
    """
    # CSV 파일 로드
    df = pd.read_csv(file_path, index_col=0)
    
    # 데이터 분할
    total_size = len(df)
    train_size = int((1 - test_size - val_size) * total_size)
    val_size = int(val_size * total_size)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    # 데이터셋 생성
    train_dataset = PerformanceDataset(
        train_df, input_timesteps, forecast_horizon, 
        target_node, target_feature, selected_features
    )
    val_dataset = PerformanceDataset(
        val_df, input_timesteps, forecast_horizon, 
        target_node, target_feature, selected_features
    )
    test_dataset = PerformanceDataset(
        test_df, input_timesteps, forecast_horizon, 
        target_node, target_feature, selected_features
    )
    
    # 데이터 정규화
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    train_dataset.X = scaler_X.fit_transform(train_dataset.X)
    val_dataset.X = scaler_X.transform(val_dataset.X)
    test_dataset.X = scaler_X.transform(test_dataset.X)
    
    train_dataset.y = scaler_y.fit_transform(train_dataset.y.reshape(-1, 1)).reshape(train_dataset.y.shape)
    val_dataset.y = scaler_y.transform(val_dataset.y.reshape(-1, 1)).reshape(val_dataset.y.shape)
    test_dataset.y = scaler_y.transform(test_dataset.y.reshape(-1, 1)).reshape(test_dataset.y.shape)
    
    # feature 이름 생성 (attention map 시각화용)
    feature_names = [
        f"{node}_{feature}" 
        for node in train_dataset.node_list 
        for feature in train_dataset.selected_features
    ]
    
    return train_dataset, val_dataset, test_dataset, scaler_X, scaler_y, feature_names

def main():
    # 설정
    config = {
        'input_timesteps': 3,
        'forecast_horizon': 2,
        'target_node': 'B2',
        'target_feature': 'ResponseTime',
        'selected_features': ['Cpu', 'Memory', 'Throughput', 'ResponseTime'],
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'patience': 10
    }
    
    # CUDA 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 데이터 준비
    train_dataset, val_dataset, test_dataset, scaler_X, scaler_y, feature_names = prepare_data(
        file_path='./preprocessed_data_slotted/89/merged_df.csv',
        input_timesteps=config['input_timesteps'],
        forecast_horizon=config['forecast_horizon'],
        target_node=config['target_node'],
        target_feature=config['target_feature'],
        selected_features=config['selected_features']
    )
    
    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # 모델 초기화
    model = SpatioTemporalTransformer(
        num_nodes=train_dataset.num_nodes,
        num_features=train_dataset.num_features,
        input_timesteps=config['input_timesteps'],
        forecast_horizon=config['forecast_horizon'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(device)
    
    # 옵티마이저와 손실 함수 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # 실험 매니저 초기화
    experiment = ExperimentManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    
    # 학습 실행
    experiment.train(num_epochs=config['num_epochs'], patience=config['patience'])
    
    # 테스트 및 결과 저장
    test_metrics = experiment.test()
    with open(experiment.save_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # 어텐션 맵 시각화
    sample_data = next(iter(test_loader))[0][:1].to(device)
    for layer_idx in range(config['num_layers']):
        experiment.visualize_attention(
            sample_data=sample_data,
            node_names=train_dataset.node_list,
            feature_names=config['selected_features'],
            layer_idx=layer_idx
        )
    
    logger.info("실험 완료!")

if __name__ == "__main__":
    main()