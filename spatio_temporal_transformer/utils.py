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