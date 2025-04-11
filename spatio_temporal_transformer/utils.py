import pandas as pd
import numpy as np
import logging
from typing import Union, Tuple
from datetime import datetime
from pathlib import Path
import joblib
import torch
from spatio_temporal_transformer_get_emb import SpatioTemporalTransformer

logger = logging.getLogger(__name__)

def find_data_by_date(dataset, target_date: Union[str, datetime]) -> int:
    """
    주어진 날짜에 해당하는 데이터의 인덱스를 찾는 함수
    
    Args:
        dataset: PerformanceDataset 객체
        target_date: 찾고자 하는 날짜 (str 또는 datetime 형식)

    Returns:
        해당 날짜에 해당하는 데이터의 인덱스
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    try:
        idx = np.where(dataset.indices == target_date)[0][0]
    except IndexError:
        # 정확한 날짜가 없을 경우 가장 가까운 날짜 찾기
        idx = np.argmin(abs(dataset.indices - target_date))
        logger.warning(f"Exact date {target_date} not found. Using closest date: {dataset.indices[idx]}")
        
    return idx

# find_data_by_date 함수를 datetime 연산이 가능하도록 수정함
def find_data_by_datetime(dataset, target_datetime: Union[str, datetime]) -> int:
    """
    주어진 날짜에 해당하는 데이터의 인덱스를 찾는 함수
    
    Args:
        dataset: PerformanceDataset 객체
        target_date: 찾고자 하는 날짜 (str 또는 datetime 형식)

    Returns:
        해당 날짜에 해당하는 데이터의 인덱스
    """
    if isinstance(target_datetime, str):
        target_datetime = pd.to_datetime(target_datetime)

    # dataset.indices를 pandas DatetimeIndex로 변환
    indices = pd.DatetimeIndex(dataset.indices)
    
    try:
        idx = indices.get_loc(target_datetime)
    except KeyError:
        # 정확한 날짜가 없을 경우 가장 가까운 날짜 찾기
        time_diffs = abs(indices - target_datetime)
        idx = time_diffs.argmin()
        logger.warning(f"Exact date {target_datetime} not found. Using closest date: {indices[idx]}")
        
    return idx


def save_scalers(scaler_X, scaler_y, save_dir):
    """
    X와 y에 대한 scaler를 저장합니다.
    
    Args:
        scaler_X: 입력 데이터의 scaler
        scaler_y: 타겟 데이터의 scaler
        save_dir: 저장할 디렉토리 경로 (Path 객체)
    """
    save_dir = Path(save_dir)
    joblib.dump(scaler_X, save_dir / 'scaler_X.joblib')
    joblib.dump(scaler_y, save_dir / 'scaler_y.joblib')

def load_scalers(save_dir):
    """
    저장된 scaler들을 불러옵니다.
    
    Args:
        save_dir: scaler가 저장된 디렉토리 경로 (Path 객체)
        
    Returns:
        scaler_X: 입력 데이터의 scaler
        scaler_y: 타겟 데이터의 scaler
    """
    save_dir = Path(save_dir)
    scaler_X = joblib.load(save_dir / 'scaler_X.joblib')
    scaler_y = joblib.load(save_dir / 'scaler_y.joblib')
    return scaler_X, scaler_y

def load_model_for_inference(model_path, device='cpu'):
    """
    저장된 모델을 불러와서 추론을 위한 상태로 준비합니다.
    
    Args:
        model_path: 저장된 모델 체크포인트 경로
        device: 모델을 불러올 디바이스 ('cpu' 또는 'cuda')
        
    Returns:
        model: 불러온 모델
        experiment_config: 모델 학습 시 사용된 설정
    """
    checkpoint = torch.load(model_path, map_location=device)
    experiment_config = checkpoint['experiment_config']
    
    # 모델 초기화
    model = SpatioTemporalTransformer(
        num_nodes=experiment_config.get('num_nodes', 5),  # 기본값 설정
        num_features=len(experiment_config.get('selected_features', ["Throughput", "ResponseTime"])),
        input_timesteps=experiment_config['input_timesteps'],
        forecast_horizon=experiment_config['forecast_horizon'],
        d_model=experiment_config['d_model'],
        nhead=experiment_config['nhead'],
        num_layers=experiment_config['num_layers'],
        dropout=experiment_config['dropout']
    ).to(device)
    
    # 저장된 가중치 불러오기
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 추론 모드로 설정
    
    return model, experiment_config

def predict(model, data, scaler_y=None):
    """
    모델을 사용해 예측을 수행합니다.
    
    Args:
        model: 불러온 모델
        data: 입력 데이터 (배치 차원 포함)
        scaler_y: 정규화에 사용된 scaler (있는 경우)
        
    Returns:
        predictions: 원래 스케일로 변환된 예측값
    """
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        
        # scaler가 제공된 경우 원래 스케일로 변환
        if scaler_y is not None:
            predictions = predictions.cpu().numpy()
            predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
            
        return predictions


def save_performance_dataset_to_csv(dataset, save_path, dataset_type):
    """
    PerformanceDataset을 CSV 형식으로 저장하는 함수
    
    Args:
        dataset: PerformanceDataset 객체
        save_path: 저장할 디렉토리 경로
        dataset_type: 'train', 'val', 'test' 중 하나
    """
    # 저장 경로 생성
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 준비
    data_list = []
    num_features = len(dataset.node_list) * len(dataset.selected_features)
    
    for i in range(len(dataset)):
        # 원본 입력 데이터 (flatten된 상태)
        X = dataset.X[i]
        # 타겟 값
        y = dataset.y[i]
        # 타임스탬프
        timestamp = dataset.indices[i]
        # 임베딩 (있는 경우)
        embedding = dataset.embeddings[i] if dataset.embeddings is not None else None
        
        # flatten된 입력 데이터를 원래 형태로 복원
        # shape: (input_timesteps, num_nodes * num_features)
        X_reshaped = X.reshape(dataset.input_timesteps, num_features)
        
        row_dict = {
            'timestamp': timestamp,
            'target_node': dataset.target_node,
            'target_feature': dataset.target_feature,
        }
        
        # 입력 데이터 추가
        for t in range(dataset.input_timesteps):
            for node_idx, node in enumerate(dataset.node_list):
                for feat_idx, feature in enumerate(dataset.selected_features):
                    col_idx = node_idx * len(dataset.selected_features) + feat_idx
                    col_name = f't{t}_{node}_{feature}'
                    row_dict[col_name] = X_reshaped[t, col_idx]
        
        # 타겟 값 추가
        for h in range(dataset.forecast_horizon):
            row_dict[f'target_t{h}'] = y[h]
        
        # 임베딩 추가 (있는 경우)
        if embedding is not None:
            for e in range(len(embedding)):
                row_dict[f'embedding_{e}'] = embedding[e]
        
        data_list.append(row_dict)
    
    # DataFrame 생성 및 저장
    df = pd.DataFrame(data_list)
    csv_path = save_dir / f'{dataset_type}_dataset_with_embeddings.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\nSaved {dataset_type} dataset to {csv_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    return df

def save_all_performance_datasets(train_loader, val_loader, test_loader, save_path):
    """
    모든 PerformanceDataset을 CSV로 저장
    """
    print("Saving PerformanceDatasets with embeddings...")
    
    train_df = save_performance_dataset_to_csv(train_loader.dataset, save_path, 'train')
    val_df = save_performance_dataset_to_csv(val_loader.dataset, save_path, 'val')
    test_df = save_performance_dataset_to_csv(test_loader.dataset, save_path, 'test')
    
    # 데이터셋 통계 출력
    print("\nDataset Statistics:")
    for name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        print(f"\n{name} Dataset:")
        print(f"- Samples: {len(df)}")
        print(f"- Timesteps: {train_loader.dataset.input_timesteps}")
        print(f"- Forecast Horizon: {train_loader.dataset.forecast_horizon}")
        if train_loader.dataset.embeddings is not None:
            print(f"- Embedding Dimension: {len(train_loader.dataset.embeddings[0])}")
