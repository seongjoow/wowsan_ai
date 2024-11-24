import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import List, Tuple, Dict
import json
from pathlib import Path

print(torch.__version__)

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
        indices = []  # 추가: 인덱스 저장용 리스트
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

            # 추가: 첫 번째 시점의 인덱스 저장
            indices.append(data.index[i])
        
        self.indices = np.array(indices)  # 클래스 변수로 저장
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
    
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("Invalid activation function")

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask,
                                            need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

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
        
        # Custom Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="relu"
            ) for _ in range(num_layers)
        ])
        
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
        
        # Initialize attention weights list
        self.attention_weights = []
        
        # Pass through each custom encoder layer and collect attention weights
        for layer in self.encoder_layers:
            x, attn_weights = layer(x)
            self.attention_weights.append(attn_weights)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # 회귀 예측
        return self.regression_head(x)
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """어텐션 가중치 추출"""
        self.attention_weights = []
        
        self.forward(x)
        
        return self.attention_weights
    
    def get_node_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        현재 상태의 노드 임베딩을 추출하는 함수
        Returns:
            node_embeddings: shape (batch_size, num_nodes, d_model)
        """
        self.eval()  # 평가 모드로 설정
        with torch.no_grad():
            batch_size = x.size(0)
            
            # 입력 데이터 전처리
            x = x.view(batch_size, self.input_timesteps, self.num_nodes, self.num_features)
            x = x.unsqueeze(-1)
            x = self.feature_embedding(x)
            
            # 인코딩 추가
            x = x + self.temporal_encoding.unsqueeze(2).unsqueeze(3)
            x = x + self.node_encoding.unsqueeze(1).unsqueeze(3)
            x = x + self.feature_encoding.unsqueeze(1).unsqueeze(2)
            
            # Transformer 입력 형태로 변환
            x = x.view(batch_size, self.input_timesteps * self.num_nodes * self.num_features, -1)
            
            # 인코더 레이어 통과
            for layer in self.encoder_layers:
                x, _ = layer(x)
            
            # 노드별 임베딩 추출
            # [batch, sequence, d_model] -> [batch, timesteps, nodes, features, d_model]
            x = x.view(batch_size, self.input_timesteps, self.num_nodes, self.num_features, -1)
            
            # 시간과 특성에 대해 평균을 내서 노드별 임베딩 얻기
            node_embeddings = x.mean(dim=(1, 3))  # [batch, nodes, d_model]
            
            return node_embeddings

class ExperimentManager:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 experiment_config: dict,
                 base_save_dir: str = "results"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.experiment_config = experiment_config
        
        # Create a descriptive folder name based on experiment configuration
        folder_name = self._create_experiment_folder_name()
        self.save_dir = Path(base_save_dir) / folder_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment configuration
        self._save_experiment_config()
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def _create_experiment_folder_name(self) -> str:
        """Create a descriptive folder name based on experiment configuration"""
        config = self.experiment_config
        # Format timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create folder name components
        components = [
            timestamp,
            f"node_{config['target_node']}",
            f"feature_{config['target_feature']}",
            f"in{config['input_timesteps']}_out{config['forecast_horizon']}",
            f"d{config['d_model']}_h{config['nhead']}_l{config['num_layers']}",
            f"bs{config['batch_size']}_lr{config['learning_rate']:.0e}"
        ]
        
        # Add selected features if specified
        if config.get('selected_features'):
            features_str = '_'.join(f[:2] for f in config['selected_features'])
            components.append(f"feat_{features_str}")
        
        return "__".join(components)
    
    def _save_experiment_config(self):
        """Save experiment configuration as JSON"""
        config_path = self.save_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.experiment_config, f, indent=4, default=str)
    
    # Rest of the ExperimentManager methods remain the same
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
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
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
        
        # Save test metrics
        metrics = calculate_metrics(all_targets, all_preds)
        metrics_path = self.save_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'experiment_config': self.experiment_config
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
                         layer_idx: int = 0,
                         test_dataset = None,
                         timestamp = None):
        """
        특정 레이어의 어텐션 맵 시각화
        
        Args:
            sample_data: 입력 데이터 텐서 (batch, timesteps * nodes * features)
            node_names: 노드 이름 리스트
            feature_names: 특성 이름 리스트
            layer_idx: 시각화할 레이어 인덱스
            test_dataset: 테스트 데이터셋
            timestamp: 시각화 대상 시점 (str 또는 datetime)
        """
        self.model.eval()
        attention_weights = self.model.get_attention_weights(sample_data)
        
        if not attention_weights or layer_idx >= len(attention_weights):
            print(f"Attention weights not available for layer {layer_idx}")
            return

        layer_attention = attention_weights[layer_idx][0]
        
        num_timesteps = self.model.input_timesteps
        num_nodes = self.model.num_nodes
        num_features = self.model.num_features
        
        labels = []
        for t in range(num_timesteps):
            for n in node_names:
                for f in feature_names:
                    labels.append(f"t{t+1}_{n}_{f}")
        
        plt.figure(figsize=(20, 16))

        # 히트맵 생성
        ax = sns.heatmap(layer_attention.detach().cpu().numpy(), 
              xticklabels=labels,
              yticklabels=labels,
              cmap='viridis')
        
        # x축 레이블을 위쪽에 표시
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        # 제목 설정 (timestamp가 있는 경우 포함)
        title = f'Attention Weights (Layer {layer_idx+1})'
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            title += f'\nTimestamp: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
        plt.title(title)

        plt.tight_layout()

        # 파일명에 timestamp 포함
        filename = f'attention_map_layer_{layer_idx+1}'
        if timestamp:
            filename += f'_{timestamp.strftime("%Y%m%d_%H%M%S")}'
        filename += '.png'
        
        plt.savefig(self.save_dir / filename)
        plt.close()

def calculate_metrics(y_true, y_pred):
    """
    시퀀스 예측에 대한 평가 메트릭 계산
    
    Args:
        y_true: shape (N, forecast_horizon)
        y_pred: shape (N, forecast_horizon)

    Returns:
        metrics: Dict with float values (not numpy types)
    """
    # MAPE 계산을 위한 헬퍼 함수
    def mape(y_true, y_pred):
        # 0으로 나누는 것을 방지하기 위해 작은 값 추가
        epsilon = 1e-10
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    metrics = {
        'mape': float(mape(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        # 각 시점별 메트릭
        'step_mape': [
            float(mape(y_true[:, i], y_pred[:, i]))
            for i in range(y_true.shape[1])
        ],
        'step_rmse': [
            float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
            for i in range(y_true.shape[1])
        ]
    }
    return metrics

def prepare_data(file_path, input_timesteps, forecast_horizon, target_node, target_feature, 
                 selected_features=None, test_size=0.2, val_size=0.1, batch_size=32):
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
    # df = pd.read_csv(file_path, index_col=0)
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date']) #  date 컬럼을 datetime 형식으로 변환
    df.set_index('date', inplace=True) # date 컬럼을 인덱스로 설정
    
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
    train_dataset.batch_size = batch_size

    val_dataset = PerformanceDataset(
        val_df, input_timesteps, forecast_horizon, 
        target_node, target_feature, selected_features
    )
    val_dataset.batch_size = batch_size

    test_dataset = PerformanceDataset(
        test_df, input_timesteps, forecast_horizon, 
        target_node, target_feature, selected_features
    )
    test_dataset.batch_size = batch_size
    
    # 데이터 정규화
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    train_dataset.X = scaler_X.fit_transform(train_dataset.X)
    val_dataset.X = scaler_X.transform(val_dataset.X)
    test_dataset.X = scaler_X.transform(test_dataset.X)
    
    train_dataset.y = scaler_y.fit_transform(train_dataset.y.reshape(-1, 1)).reshape(train_dataset.y.shape)
    val_dataset.y = scaler_y.transform(val_dataset.y.reshape(-1, 1)).reshape(val_dataset.y.shape)
    test_dataset.y = scaler_y.transform(test_dataset.y.reshape(-1, 1)).reshape(test_dataset.y.shape)
    
    # node names and feature names for attention map visualization
    node_names = train_dataset.node_list
    feature_names = train_dataset.selected_features
    
    return train_dataset, val_dataset, test_dataset, scaler_X, scaler_y, node_names, feature_names

if __name__ == "__main__":
    from utils import find_data_by_date

    # Parameters
    experiment_config = {
        'file_path': './preprocessed_data_sttransformer/176/merged_df.csv',
        'input_timesteps': 10,
        'forecast_horizon': 2,
        'target_node': "B2",
        'target_feature': "ResponseTime",
        'selected_features': ["Throughput", "ResponseTime"],
        'batch_size': 32,
        'num_epochs': 100,
        'patience': 10,
        'learning_rate': 1e-3,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dropout': 0.1
    }
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, scaler_X, scaler_y, node_names, feature_names = prepare_data(
        experiment_config['file_path'],
        experiment_config['input_timesteps'],
        experiment_config['forecast_horizon'],
        experiment_config['target_node'],
        experiment_config['target_feature'],
        experiment_config['selected_features'],
        batch_size=experiment_config['batch_size']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=experiment_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=experiment_config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=experiment_config['batch_size'])
    
    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalTransformer(
        num_nodes=train_dataset.num_nodes,
        num_features=train_dataset.num_features,
        input_timesteps=experiment_config['input_timesteps'],
        forecast_horizon=experiment_config['forecast_horizon'],
        d_model=experiment_config['d_model'],
        nhead=experiment_config['nhead'],
        num_layers=experiment_config['num_layers'],
        dropout=experiment_config['dropout']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Initialize experiment manager
    experiment = ExperimentManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        experiment_config=experiment_config,
        base_save_dir="results"
    )
    
    # Train the model
    experiment.train(num_epochs=experiment_config['num_epochs'], 
                    patience=experiment_config['patience'])
    
    # Test the model
    metrics = experiment.test()
    print("Test Metrics:", metrics)
    
    # Visualize attention weights

    # testset의 첫 번째 배치 데이터로 시각화
    # sample_data, _ = next(iter(test_loader))
    # sample_data = sample_data.to(device)

    # 특정 데이터에 대한 어텐션 맵 시각화
    # Visualize attention weights for a specific date
    target_date = "2024-11-10 20:04:30"  # 원하는 날짜 지정
    data_idx = find_data_by_date(test_dataset, target_date)
    sample_data = test_dataset[data_idx][0].unsqueeze(0).to(device)

    # 실제 사용된 날짜 (정확한 날짜가 없는 경우 가장 가까운 날짜가 사용됨)
    actual_date = test_dataset.indices[data_idx]

    # 전체 레이어에 대한 어텐션 맵 시각화
    num_layers = len(model.encoder_layers)
    for i in range(num_layers):
        experiment.visualize_attention(
            sample_data,
            node_names,
            feature_names,
            layer_idx=i,
            test_dataset=test_dataset,
            timestamp=actual_date  # 실제 사용된 날짜
        )
