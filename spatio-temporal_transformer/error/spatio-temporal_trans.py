import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import logging
import json
from pathlib import Path
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SpatioTemporalDataset(Dataset):
    def __init__(self, 
                 data: np.ndarray,
                 input_timesteps: int,
                 num_nodes: int,
                 target_node: int,
                 scaler: StandardScaler = None,
                 train: bool = True):
        """
        Args:
            data: Raw data array of shape (num_samples, num_timesteps * num_nodes)
            input_timesteps: Number of input timesteps
            num_nodes: Number of nodes in the network
            target_node: Index of the target node for prediction
            scaler: StandardScaler for feature normalization
            train: Whether this is training data (for fitting scaler)
        """
        self.input_timesteps = input_timesteps
        self.num_nodes = num_nodes
        self.target_node = target_node
        
        # 데이터 정규화
        if scaler is None and train:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(data)
        
        # 데이터 재구성: (samples, timesteps, nodes)
        self.data = self.data.reshape(-1, input_timesteps + 1, num_nodes)
        
        # 입력과 타겟 분리
        self.X = torch.FloatTensor(self.data[:, :-1, :])
        self.y = torch.FloatTensor(self.data[:, -1, target_node])
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SpatioTemporalTransformer(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 num_timesteps: int,
                 d_model: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        
        # Feature embedding
        self.value_embedding = nn.Linear(1, d_model)
        
        # Positional encodings
        self.temporal_encoding = nn.Parameter(torch.zeros(1, num_timesteps, d_model))
        self.node_encoding = nn.Parameter(torch.zeros(1, num_nodes, d_model))

        self.attention_weights = None
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output layer
        self.regression_head = nn.Sequential(
            nn.Linear(d_model * num_nodes * num_timesteps, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, timesteps, nodes)
        batch_size = x.size(0)
        
        # Reshape for node-wise embedding
        x = x.view(batch_size, self.num_timesteps, self.num_nodes, 1)
        x = self.value_embedding(x)  # (batch, timesteps, nodes, d_model)
        
        # Add positional encodings
        x = x + self.temporal_encoding.unsqueeze(2)
        x = x + self.node_encoding.unsqueeze(1)
        
        # Reshape for transformer input
        x = x.view(batch_size, self.num_timesteps * self.num_nodes, -1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Flatten and predict
        x = x.reshape(batch_size, -1)
        return self.regression_head(x).squeeze()
    
    def get_attention_weights(self, x):
        """
        입력 데이터에 대한 attention weights 반환
        Returns:
            attention_weights: List[Tensor], 각 레이어의 attention weights
            shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
        """
        self.attention_weights = []
        
        # 기존 forward 로직과 유사하게 진행
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_timesteps, self.num_nodes, 1)
        x = self.value_embedding(x)
        x = x + self.temporal_encoding.unsqueeze(2)
        x = x + self.node_encoding.unsqueeze(1)
        x = x.view(batch_size, self.num_timesteps * self.num_nodes, -1)
        
        # 각 transformer 레이어의 attention weights 수집
        for layer in self.transformer_layers:
            # TransformerEncoderLayer의 self-attention 부분 접근
            attention_output = layer.self_attn(
                x, x, x,
                attn_mask=None,
                need_weights=True
            )
            x = layer.norm1(x + layer.dropout1(attention_output[0]))
            x = layer.norm2(x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
            
            # attention weights 저장
            self.attention_weights.append(attention_output[1])
        
        return self.attention_weights
    
def visualize_attention_patterns(
    model: SpatioTemporalTransformer,
    sample_data: torch.Tensor,
    node_names: List[str],
    save_path: str = None,
    layer_idx: int = 0,
    head_idx: int = 0
):
    """
    Attention 패턴을 시각화하는 함수
    
    Args:
        model: 학습된 SpatioTemporalTransformer 모델
        sample_data: 시각화할 입력 데이터 (batch_size, timesteps, nodes)
        node_names: 노드 이름 리스트
        save_path: 결과 저장 경로
        layer_idx: 시각화할 레이어 인덱스
        head_idx: 시각화할 어텐션 헤드 인덱스
    """
    model.eval()
    with torch.no_grad():
        attention_weights = model.get_attention_weights(sample_data)
    
    # 지정된 레이어와 헤드의 attention weights 추출
    attn_matrix = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    
    # 시간-노드 레이블 생성
    num_timesteps = model.num_timesteps
    num_nodes = model.num_nodes
    labels = [f't{t+1}_{node}' for t in range(num_timesteps) for node in node_names]
    
    # Attention 맵 시각화
    plt.figure(figsize=(15, 12))
    
    # 메인 히트맵
    plt.subplot(2, 2, 1)
    sns.heatmap(attn_matrix, 
                xticklabels=labels,
                yticklabels=labels,
                cmap='YlOrRd',
                center=0.5)
    plt.title(f'Full Attention Pattern (Layer {layer_idx+1}, Head {head_idx+1})')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    # 시간적 패턴 (같은 노드 간의 attention)
    temporal_pattern = np.zeros((num_timesteps, num_timesteps))
    for i in range(num_nodes):
        for t1 in range(num_timesteps):
            for t2 in range(num_timesteps):
                idx1 = t1 * num_nodes + i
                idx2 = t2 * num_nodes + i
                temporal_pattern[t1, t2] = attn_matrix[idx1, idx2]
    
    plt.subplot(2, 2, 2)
    sns.heatmap(temporal_pattern,
                xticklabels=[f't{i+1}' for i in range(num_timesteps)],
                yticklabels=[f't{i+1}' for i in range(num_timesteps)],
                cmap='YlOrRd',
                center=0.5)
    plt.title('Temporal Attention Pattern\n(Same Node)')
    
    # 공간적 패턴 (같은 시점 내 노드 간의 attention)
    spatial_pattern = np.zeros((num_nodes, num_nodes))
    for t in range(num_timesteps):
        for n1 in range(num_nodes):
            for n2 in range(num_nodes):
                idx1 = t * num_nodes + n1
                idx2 = t * num_nodes + n2
                spatial_pattern[n1, n2] = attn_matrix[idx1, idx2]
    
    plt.subplot(2, 2, 3)
    sns.heatmap(spatial_pattern,
                xticklabels=node_names,
                yticklabels=node_names,
                cmap='YlOrRd',
                center=0.5)
    plt.title('Spatial Attention Pattern\n(Same Timestep)')
    
    # Average attention distribution
    plt.subplot(2, 2, 4)
    avg_attention = attn_matrix.mean(axis=0)
    plt.bar(range(len(labels)), avg_attention)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.title('Average Attention Distribution')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

class ExperimentTracker:
    def __init__(self, exp_name: str, save_dir: str = "experiments"):
        self.exp_name = exp_name
        self.save_dir = Path(save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_metrics': {},
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
        
    def save_config(self, config: Dict):
        with open(self.save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=4)
    
    def update(self, epoch: int, train_loss: float, val_loss: float):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        if val_loss < self.history['best_val_loss']:
            self.history['best_val_loss'] = val_loss
            self.history['best_epoch'] = epoch
            return True
        return False
    
    def save_model(self, model: nn.Module, filename: str):
        torch.save(model.state_dict(), self.save_dir / filename)
    
    def save_test_metrics(self, metrics: Dict):
        self.history['test_metrics'] = metrics
        with open(self.save_dir / "test_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training History - {self.exp_name}')
        plt.legend()
        plt.savefig(self.save_dir / "loss_history.png")
        plt.close()

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: str) -> float:
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model: nn.Module,
             val_loader: DataLoader,
             criterion: nn.Module,
             device: str) -> float:
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def test_model(model: nn.Module,
               test_loader: DataLoader,
               device: str) -> Dict:
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return {
        'mse': mean_squared_error(actuals, predictions),
        'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
        'mae': mean_absolute_error(actuals, predictions),
        'r2': r2_score(actuals, predictions)
    }

def run_experiment(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    config: Dict,
    exp_name: str = "experiment"
) -> Tuple[Dict, ExperimentTracker]:
    """
    실험 전체 과정을 실행하는 메인 함수
    
    Args:
        train_data: 학습 데이터
        val_data: 검증 데이터
        test_data: 테스트 데이터
        config: 실험 설정
        exp_name: 실험 이름
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tracker = ExperimentTracker(exp_name)
    tracker.save_config(config)
    
    # 데이터셋 준비
    train_dataset = SpatioTemporalDataset(
        train_data,
        config['input_timesteps'],
        config['num_nodes'],
        config['target_node'],
        train=True
    )
    
    val_dataset = SpatioTemporalDataset(
        val_data,
        config['input_timesteps'],
        config['num_nodes'],
        config['target_node'],
        scaler=train_dataset.scaler,
        train=False
    )
    
    test_dataset = SpatioTemporalDataset(
        test_data,
        config['input_timesteps'],
        config['num_nodes'],
        config['target_node'],
        scaler=train_dataset.scaler,
        train=False
    )
    
    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # 모델 초기화
    model = SpatioTemporalTransformer(
        num_nodes=config['num_nodes'],
        num_timesteps=config['input_timesteps'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # 학습 설정
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 학습 루프
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if tracker.update(epoch, train_loss, val_loss):
            tracker.save_model(model, "best_model.pth")
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{config['epochs']}")
            logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 최종 평가
    model.load_state_dict(torch.load(tracker.save_dir / "best_model.pth"))
    test_metrics = test_model(model, test_loader, device)
    tracker.save_test_metrics(test_metrics)
    tracker.plot_losses()
    
    return test_metrics, tracker


# 실험 실행 예시
if __name__ == "__main__":
    # 설정 예시
    config = {
        'input_timesteps': 3,  # t1, t2, t3
        'num_nodes': 4,        # b1, b2, b3, b4
        'target_node': 1,      # b2 예측
        'batch_size': 32,
        'd_model': 64,
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'epochs': 100
    }
    
    # 데이터 생성 예시 (실제 데이터로 교체 필요)
    num_samples = 1000
    total_features = config['input_timesteps'] * config['num_nodes']
    
    train_data = np.random.randn(num_samples, total_features + config['num_nodes'])
    val_data = np.random.randn(num_samples//4, total_features + config['num_nodes'])
    test_data = np.random.randn(num_samples//4, total_features + config['num_nodes'])
    
    # 노드 이름 설정
    node_names = ['b1', 'b2', 'b3', 'b4']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 실험 실행
    metrics, tracker = run_experiment(
        train_data,
        val_data,
        test_data,
        config,
        "spatiotemporal_experiment"
    )
    
    print("\nTest Metrics:")
    print(json.dumps(metrics, indent=4))
    
    # 학습된 모델 로드
    model = SpatioTemporalTransformer(
        num_nodes=config['num_nodes'],
        num_timesteps=config['input_timesteps'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(torch.load(f"experiments/spatiotemporal_experiment/best_model.pth"))
    
    # 테스트 데이터 하나 선택하여 attention 시각화
    test_dataset = SpatioTemporalDataset(
        test_data,
        config['input_timesteps'],
        config['num_nodes'],
        config['target_node'],
        train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1)  # batch_size=1 for visualization
    sample_data = next(iter(test_loader))[0]  # 첫 번째 배치의 데이터만 가져옴
    
    # Attention 시각화
    visualize_attention_patterns(
        model,
        sample_data,
        node_names,
        save_path="experiments/spatiotemporal_experiment/attention_visualization.png"
    )
    
    logging.info(f"실험 완료!")
    logging.info(f"실험 결과: experiments/spatiotemporal_experiment/")