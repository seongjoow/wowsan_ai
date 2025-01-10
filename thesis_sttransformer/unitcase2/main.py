import json
# import numpy as np
import pandas as pd
# from dateutil import parser
import os
import sys
import torch
from pathlib import Path
import json
import gc
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(parent_dir, 'preprocess_sttransformer'))
sys.path.append(os.path.join(parent_dir, 'spatio_temporal_transformer'))
sys.path.append('../../')

import preprocess_hop_log, merge_hop_tick, aggregate_rows_by_time
# from spatio_temporal_transformer import prepare_data, ExperimentManager, SpatioTemporalTransformer
import spatio_temporal_transformer_get_emb as stt
# from spatio_temporal_transformer_use_emb import prepare_data, ExperimentManager, SpatioTemporalTransformer, PerformanceDataset
import spatio_temporal_transformer_use_emb as stt_emb
import utils

def create_merged_df(dir_name, num_brokers, broker_port):
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    # 상위 폴더의 상위 폴더 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    
    # 각 폴더 경로 설정
    path_hoplog = os.path.join(parent_dir, 'raw_data', 'hopLogger', str(dir_name), '')
    path_ticklog = os.path.join(parent_dir, 'raw_data', 'tickLogger', str(dir_name), '')
    # path_mergeddf = os.path.join(parent_dir, 'preprocessed_data_sttransformer', str(dir_name), '')

    # path_hoplog = f'../../raw_data/hopLogger/{dir_name}/'
    # path_ticklog = f'../../raw_data/tickLogger/{dir_name}/'
    path_mergeddf = f'./thesis_sttransformer/unitcase2/preprocessed_data/{dir_name}/{broker_port}/'

    ''' Hop Log 전처리 '''
    # json 파일을 열고 각 줄을 개별적으로 파싱
    data = []
    with open(path_hoplog + f'{broker_port}.json', 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # 데이터를 DataFrame으로 변환
    hop_df_raw = pd.DataFrame(data)

    # hop_df_raw 전처리
    hop_df = preprocess_hop_log.parse_performance_info(num_brokers, hop_df_raw)

    # hop_df 저장
    if not os.path.exists(path_mergeddf):
        os.makedirs(path_mergeddf)
    hop_df.to_csv(path_mergeddf + 'hop_df_dup.csv', index=False)

    # 1초에 2개 이상의 행이 존재할 때 0을 제외한 값들끼리의 평균값을 구해서 하나의 date에는 하나의 행만 남김
    hop_df = aggregate_rows_by_time.handle_grouped_duplicate_time(hop_df, method='mean')
    
    # hop_df 저장
    hop_df.to_csv(path_mergeddf + 'hop_df_agg.csv', index=False)


    ''' Tick Log 전처리 '''
    data = []
    with open(path_ticklog + f'{broker_port}_tick.json', 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # 데이터를 DataFrame으로 변환
    tick_df_raw = pd.DataFrame(data)

    # 여러 컬럼 한 번에 type 변경
    tick_df = tick_df_raw.astype({
        'AverageQueueTime': float, 'AverageServiceTime': float,
        'AverageThroughput': float, 'InterArrivalTime': float,
        'Memory': float, 'MessageCount': float,
        'QueueLength': float, 'QueueTime': float,
        'ResponseTime': float, 'ServiceTime': float,
        'Throughput': float, 'TotalArrival Time': float
    })

    # tick_df 저장
    tick_df.to_csv(path_mergeddf + 'tick_df.csv', index=False)


    ''' Hop and Tick Data 병합 '''
    # Merge hop and tick dataframes
    merged_df = merge_hop_tick.merge_dataframes(hop_df, tick_df, broker_port)
    merged_df['Bottleneck'] = merged_df['Bottleneck'].astype('float64')

    # 학습에 불필요한 컬럼 제거
    drop_columns = [
        'level', 'msg', 'AverageQueueTime',
        'AverageServiceTime', 'AverageThroughput',
        'MessageCount',
        # 'Node', 'TotalArrivalTime', 'Timestamp'
    ]
    merged_df.drop(columns=drop_columns, inplace=True)
    merged_df = merged_df.rename(columns={'time': 'date'})

    # 컬럼 재정렬
    new_order = ['date', 'Bottleneck',
                #  'HopCount', 
                #  'B1_Timestamp', 'B1_BrokerId',
                 'B1_Cpu', 'B1_Memory', 'B1_InterArrivalTime', 'B1_QueueLength', 
                 'B1_QueueTime', 'B1_ServiceTime', 'B1_ResponseTime', 'B1_Throughput',
                 #  'B2_Timestamp',  'B2_BrokerId',
                 'B2_Cpu', 'B2_Memory', 'B2_InterArrivalTime', 'B2_QueueLength',  
                 'B2_QueueTime', 'B2_ServiceTime', 'B2_ResponseTime', 'B2_Throughput',
                 #  'B3_Timestamp', 'B3_BrokerId',
                 'B3_Cpu', 'B3_Memory', 'B3_InterArrivalTime', 'B3_QueueLength', 
                 'B3_QueueTime', 'B3_ServiceTime', 'B3_ResponseTime', 'B3_Throughput',
                 #  'B4_Timestamp', 'B4_BrokerId',
                 'B4_Cpu', 'B4_Memory', 'B4_InterArrivalTime', 'B4_QueueLength', 
                 'B4_QueueTime', 'B4_ServiceTime', 'B4_ResponseTime', 'B4_Throughput',
                 #  'B5_Timestamp', 'B5_BrokerId',
                 'B5_Cpu', 'B5_Memory', 'B5_InterArrivalTime', 'B5_QueueLength', 
                 'B5_QueueTime', 'B5_ServiceTime', 'B5_ResponseTime', 'B5_Throughput']

    merged_df = merged_df[new_order]

    # 불필요한 컬럼 제거
    merged_df.drop(columns=['Bottleneck'], inplace=True)

    # NaN 값을 0으로 대체
    merged_df.fillna(0, inplace=True)

    # 최종 데이터 저장
    merged_df.to_csv(path_mergeddf + 'merged_df.csv', 
                    date_format='%Y-%m-%d %H:%M:%S', 
                    index=False)

    return merged_df

def prepare_experiment_stt(timestamp, file_name, simulation_id, port, input_timesteps=10, forecast_horizon=2):
    """
    실험을 위한 설정과 데이터를 준비합니다.
    
    Args:
        file_name: 데이터 파일명 (e.g., 'merged_0to1')
        simulation_id: 시뮬레이션 ID (e.g., 210)
        port: 브로커 포트 번호
        input_timesteps: 입력 시퀀스 길이
        forecast_horizon: 예측 기간
    
    Returns:
        experiment_config: 실험 설정
        train_loader, val_loader, test_loader: 데이터 로더
        scaler_X, scaler_y: 데이터 스케일러
        node_names, feature_names: 노드와 특성 이름
        save_dir: 실험 결과가 저장될 경로
    """
    # 실험 설정 준비
    experiment_config = {
        'file_path': f'./thesis_sttransformer/unitcase2/preprocessed_data/{simulation_id}/{port}/{file_name}.csv',
        'input_timesteps': input_timesteps,
        'forecast_horizon': forecast_horizon,
        'target_node': f"B{port - 50000}",
        'target_feature': "ResponseTime",
        'selected_features': ["Throughput", "ResponseTime"],
        'batch_size': 32,
        'num_epochs': 100,
        'patience': 10,
        'learning_rate': 1e-3,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dropout': 0.1,
        'broker_port': port
    }

    # 현재 시간을 기반으로 폴더명 생성
    # timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}__node_B{port - 50000}__in{input_timesteps}_out{forecast_horizon}"

    # 결과 저장 디렉토리 생성
    save_dir = Path(f"./thesis_sttransformer/unitcase2/results/{port}/{folder_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 준비
    train_dataset, val_dataset, test_dataset, scaler_X, scaler_y, node_names, feature_names = stt.prepare_data(
        experiment_config['file_path'],
        experiment_config['input_timesteps'],
        experiment_config['forecast_horizon'],
        experiment_config['target_node'],
        experiment_config['target_feature'],
        experiment_config['selected_features'],
        batch_size=experiment_config['batch_size']
    )

    # 스케일러 저장
    utils.save_scalers(scaler_X, scaler_y, save_dir)
    print(f"Saved scalers to {save_dir}")

    # 실험 설정 파일도 저장
    with open(save_dir / "experiment_config.json", 'w') as f:
        json.dump(experiment_config, f, indent=4)

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=experiment_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=experiment_config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=experiment_config['batch_size'])

    return (experiment_config, train_loader, val_loader, test_loader, 
            scaler_X, scaler_y, node_names, feature_names, save_dir)

def prepare_experiment_stt_emb(timestamp, file_name, simulation_id, port, input_timesteps=10, forecast_horizon=2):
    """
    실험을 위한 설정과 데이터를 준비합니다.
    
    Args:
        file_name: 데이터 파일명 (e.g., 'merged_0to1')
        simulation_id: 시뮬레이션 ID (e.g., 210)
        port: 브로커 포트 번호
        input_timesteps: 입력 시퀀스 길이
        forecast_horizon: 예측 기간
    
    Returns:
        experiment_config: 실험 설정
        train_loader, val_loader, test_loader: 데이터 로더
        scaler_X, scaler_y: 데이터 스케일러
        node_names, feature_names: 노드와 특성 이름
        save_dir: 실험 결과가 저장될 경로
    """
    # 실험 설정 준비
    experiment_config = {
        'file_path': f'./thesis_sttransformer/unitcase2/preprocessed_data/{simulation_id}/{port}/{file_name}.csv',
        'input_timesteps': input_timesteps,
        'forecast_horizon': forecast_horizon,
        'target_node': f"B{port - 50000}",
        'target_feature': "ResponseTime",
        'selected_features': ["Throughput", "ResponseTime"],
        'batch_size': 32,
        'num_epochs': 100,
        'patience': 10,
        'learning_rate': 1e-3,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dropout': 0.1,
        'broker_port': port
    }

    # 현재 시간을 기반으로 폴더명 생성
    # timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}__node_B{port - 50000}__in{input_timesteps}_out{forecast_horizon}"

    # 결과 저장 디렉토리 생성
    save_dir = Path(f"./thesis_sttransformer/unitcase2/results/{port}/{folder_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 준비
    train_dataset, val_dataset, test_dataset, scaler_X, scaler_y, node_names, feature_names = stt_emb.prepare_data(
        experiment_config['file_path'],
        experiment_config['input_timesteps'],
        experiment_config['forecast_horizon'],
        experiment_config['target_node'],
        experiment_config['target_feature'],
        experiment_config['selected_features'],
        batch_size=experiment_config['batch_size']
    )

    # 스케일러 저장
    utils.save_scalers(scaler_X, scaler_y, save_dir)
    print(f"Saved scalers to {save_dir}")

    # 실험 설정 파일도 저장
    with open(save_dir / "experiment_config.json", 'w') as f:
        json.dump(experiment_config, f, indent=4)

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=experiment_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=experiment_config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=experiment_config['batch_size'])

    return (experiment_config, train_loader, val_loader, test_loader, 
            scaler_X, scaler_y, node_names, feature_names, save_dir)

def train_model_stt(experiment_config, train_loader, val_loader, test_loader, device, save_dir):
    """
    모델을 초기화하고 학습을 수행합니다.
    
    Args:
        experiment_config: 실험 설정
        train_loader, val_loader, test_loader: 데이터 로더
        device: 학습 장치 (cuda/cpu)
        save_dir: 결과 저장 경로
    
    Returns:
        experiment: 학습된 실험 관리자 객체
    """
    model = stt.SpatioTemporalTransformer(
        num_nodes=train_loader.dataset.num_nodes,
        num_features=train_loader.dataset.num_features,
        input_timesteps=experiment_config['input_timesteps'],
        forecast_horizon=experiment_config['forecast_horizon'],
        d_model=experiment_config['d_model'],
        nhead=experiment_config['nhead'],
        num_layers=experiment_config['num_layers'],
        dropout=experiment_config['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config['learning_rate'])
    criterion = nn.MSELoss()

    experiment = stt.ExperimentManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        experiment_config=experiment_config,
        save_dir=save_dir
        # base_save_dir=f"./thesis_sttransformer/unitcase2/results/{experiment_config['broker_port']}"
    )

    print(f"\n=== Training model for broker {experiment_config['broker_port']} ===")
    experiment.train(num_epochs=experiment_config['num_epochs'], 
                    patience=experiment_config['patience'])

    return experiment

def train_model_stt_emb(experiment_config, train_loader, val_loader, test_loader, device, save_dir):
    """
    모델을 초기화하고 학습을 수행합니다.
    
    Args:
        experiment_config: 실험 설정
        train_loader, val_loader, test_loader: 데이터 로더
        device: 학습 장치 (cuda/cpu)
        save_dir: 결과 저장 경로
    
    Returns:
        experiment: 학습된 실험 관리자 객체
    """

    # d_embedding = train_loader.dataset.embeddings.shape[1]  # 임베딩 차원
    # 임베딩 차원 확인
    emb_df = pd.read_csv(Path(save_dir) / "node_embeddings.csv", index_col=0)
    d_embedding = emb_df.shape[1]
    print(f"Actual embedding dimension: {d_embedding}")  # 디버깅용

    model = stt_emb.SpatioTemporalTransformer(
        num_nodes=train_loader.dataset.num_nodes,
        num_features=train_loader.dataset.num_features,
        input_timesteps=experiment_config['input_timesteps'],
        forecast_horizon=experiment_config['forecast_horizon'],
        d_embedding=d_embedding,  # 임베딩 차원
        d_model=experiment_config['d_model'],
        nhead=experiment_config['nhead'],
        num_layers=experiment_config['num_layers'],
        dropout=experiment_config['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config['learning_rate'])
    criterion = nn.MSELoss()

    experiment = stt_emb.ExperimentManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        experiment_config=experiment_config,
        save_dir=save_dir
        # base_save_dir=f"./thesis_sttransformer/unitcase2/results/{experiment_config['broker_port']}"
    )

    print(f"\n=== Training model for broker {experiment_config['broker_port']} with embeddings ===")
    experiment.train(num_epochs=experiment_config['num_epochs'], 
                    patience=experiment_config['patience'])

    return experiment

def test_model(experiment, device, save_dir):
    """
    학습된 모델을 테스트합니다.
    
    Args:
        experiment: 학습된 실험 관리자 객체
        device: 학습 장치 (cuda/cpu)
        save_dir: 결과 저장 경로
    
    Returns:
        metrics: 테스트 결과 지표
    """
    # 모델 테스트
    print(f"\n=== Testing model for broker {experiment.experiment_config['broker_port']} ===")
    metrics = experiment.test()
    print(f"Test Metrics:", json.dumps(metrics, indent=2))

    # # 모델 로딩 테스트
    # print(f"\n=== Testing model loading ===")
    # model_path = experiment.save_dir / "best_model.pth"
    # loaded_model, _ = utils.load_model_for_inference(model_path, device)

    return metrics

def extract_embeddings(port, simulation_id, device, experiment_path):
    """
    저장된 모델을 불러와서 두 번째 시간대 데이터의 각 데이터 포인트(윈도우)에 대한 임베딩을 추출합니다.
    
    Args:
        port: 브로커 포트 번호
        simulation_id: 시뮬레이션 ID
        device: 학습 장치 (cuda/cpu)
        experiment_path: 실험 결과가 저장된 경로
    
    Returns:
        embeddings_df: 각 데이터 포인트의 임베딩이 저장된 DataFrame
                        index: 데이터 포인트의 타임스탬프
                        columns: 표현 벡터의 각 차원 (d_model개)
    """
    # 모델 설정 불러오기
    config_path = experiment_path / "experiment_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 저장된 scaler 불러오기
    scaler_X, _ = utils.load_scalers(experiment_path)
    
    # 두 번째 시간대 데이터 준비
    df = pd.read_csv(f'./thesis_sttransformer/unitcase2/preprocessed_data/{simulation_id}/{port}/merged_1to2.csv')
    df['date'] = pd.to_datetime(df['date'])  # date 열을 datetime으로 변환
    df.set_index('date', inplace=True)  # date를 인덱스로 설정

    dataset = stt_emb.PerformanceDataset(
        df,
        config['input_timesteps'],
        config['forecast_horizon'],
        config['target_node'],
        config['target_feature'],
        config['selected_features']
    )
    
    # 입력 데이터 정규화
    normalized_X = scaler_X.transform(dataset.X)
    
    # 모델 불러오기
    model_path = experiment_path / "best_model_stt.pth"
    model, _ = utils.load_model_for_inference(model_path, device)
    # print(f"\n=== Loaded model for broker {port} ===")
    
    # 임베딩 추출
    embeddings = []
    timestamps = []
    # window_end_times = dataset.indices
    # print("window_end_times[0]:", window_end_times[0])
    
    for i in range(len(normalized_X)):
        # 하나의 데이터 포인트에 대한 처리
        input_data = torch.from_numpy(normalized_X[i:i+1]).float().to(device)
        emb = model.get_node_embedding(input_data)  # [1, d_model]
        embeddings.append(emb.cpu().numpy())
        timestamps.append(dataset.indices[i])
    
    # 결과를 DataFrame으로 변환
    embeddings = np.vstack([emb for emb in embeddings])  # shape: [num_windows, embedding_dim]
    print(f"Embeddings shape before creating DataFrame: {embeddings.shape}")  # 디버깅용

    # DataFrame 생성 (각 행이 하나의 데이터 포인트에 대한 임베딩)
    columns = [f'emb_{i}' for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(
        embeddings,
        columns=columns,
        index=timestamps
    )
    
    # 저장
    # save_path = f'./thesis_sttransformer/unitcase2/embeddings/{simulation_id}/{port}'
    save_path = experiment_path
    Path(save_path).mkdir(parents=True, exist_ok=True)
    embeddings_df.to_csv(f'{save_path}/node_embeddings.csv')
                        #  date_format='%Y-%m-%d %H:%M:%S')

    # print(f"\nExtracted node embeddings for broker {port}:")
    # print(f"Shape: {embeddings_df.shape}")
    # print(f"Time range: {embeddings_df.index[0]} to {embeddings_df.index[-1]}")
    
    return embeddings_df


if __name__ == "__main__":
    # DIR_NAME = 210
    # NUM_BROKERS = 5
    # BROKER_PORT = 50003
    # merged_df = process_data(DIR_NAME, NUM_BROKERS, BROKER_PORT)
    # print("Data processing completed successfully!")
    # print(f"DataFrame shapes: {merged_df.shape}")

    def clear_memory():
        """메모리 정리 함수"""
        # PyTorch CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 가비지 컬렉터 실행
        gc.collect()

    broker_ports = [50003, 50004, 50005]
    simulation_id = 210
    firsthalf_dfs = {}
    secondhalf_dfs = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for port in broker_ports:
        # processed_dfs[port] = process_data(simulation_id, 5, port)
        # print(f"Processed data for broker {port} successfully! Shape: {processed_dfs[port].shape}")

        # 데이터 전처리
        full_df = create_merged_df(simulation_id, 5, port)

        # date 컬럼을 datetime으로 변환
        full_df['date'] = pd.to_datetime(full_df['date'])
        
        # 첫 1시간 데이터와 나머지 데이터 분리
        start_time = full_df['date'].min()
        end_time = start_time + pd.Timedelta(hours=1)

         # 첫 1시간 데이터 필터링
        firsthalf_dfs[port] = full_df[(full_df['date'] >= start_time) & (full_df['date'] < end_time)]
        # 나머지 데이터 필터링
        secondhalf_dfs[port] = full_df[full_df['date'] >= end_time]

        # 필터링된 데이터 저장
        filtered_path = f'./thesis_sttransformer/unitcase2/preprocessed_data/{simulation_id}/{port}/merged_0to1.csv'
        firsthalf_dfs[port].to_csv(filtered_path, index=False)
        # 각 데이터 저장
        base_path = f'./thesis_sttransformer/unitcase2/preprocessed_data/{simulation_id}/{port}'
        # 첫 1시간 데이터 저장
        firsthalf_dfs[port].to_csv(f'{base_path}/merged_0to1.csv', index=False)
        # 나머지 데이터 저장
        secondhalf_dfs[port].to_csv(f'{base_path}/merged_1to2.csv', index=False)
        # 전체 데이터는 이미 저장되어 있음 (merged_df.csv)
        
        print(f"\nProcessed data for broker {port}:")
        print(f"Total data shape: {full_df.shape}")
        print(f"First half hour data shape: {firsthalf_dfs[port].shape}")
        print(f"Second half hour data shape: {secondhalf_dfs[port].shape}")
        print(f"First half hour time range: {firsthalf_dfs[port]['date'].min()} to {firsthalf_dfs[port]['date'].max()}")
        print(f"Second half hour time range: {secondhalf_dfs[port]['date'].min()} to {secondhalf_dfs[port]['date'].max()}")
    
        print("\nData files saved:")
        base_path = f'./thesis_sttransformer/unitcase2/preprocessed_data/{simulation_id}/{port}'
        print(f"Broker {port}:")
        print(f"- Full data: {base_path}/merged_df.csv")
        print(f"- First half hour data: {base_path}/merged_0to1.csv")
        print(f"- Second half hour data: {base_path}/merged_1to2.csv")


    experiment_paths = {}  # 각 브로커별 실험 경로를 저장할 딕셔너리
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


    ''' 00:00:00 ~ 01:00:00 데이터 (성능 지표) 사용 '''

    # 각 브로커별 모델 학습 및 테스트
    for port in broker_ports:
        try:
            print(f"\n=== Starting experiment for broker {port} ===")
            
            # 실험 준비 (스케일러는 이 단계에서 저장됨)
            (experiment_config, train_loader, val_loader, test_loader,
            scaler_X, scaler_y, node_names, feature_names, save_dir) = prepare_experiment_stt(timestamp, "merged_0to1", simulation_id, port)
            
            # 실험 경로 저장
            experiment_paths[port] = save_dir

            # 모델 학습
            experiment = train_model_stt(experiment_config, train_loader, val_loader, test_loader, device, save_dir)

            # 모델 테스트
            metrics = test_model(experiment, device, save_dir)
            
            print(f"\nExperiment for broker {port} completed successfully!")
            print(f"Results saved in: {experiment.save_dir}")
            
            # 메모리 정리
            del experiment
            clear_memory()
            
        except Exception as e:
            print(f"Error processing broker {port}: {str(e)}")
            clear_memory()
            continue

    print("\nAll experiments completed successfully!")


    ''' 01:00:00 ~ 02:00:00 데이터 (성능 지표, 임베딩) 사용 '''

    # 학습된 모델로 두 번째 시간대 데이터의 노드 임베딩 추출
    print("\nExtracting node embeddings for second half data...")
    embeddings = {}
    for port in broker_ports:
        try:
            if port not in experiment_paths:
                print(f"No experiment path found for broker {port}, skipping...")
                continue
            
            # print(f"Using experiment path: {experiment_paths[port]}")
            embeddings[port] = extract_embeddings(port, simulation_id, device, experiment_path=experiment_paths[port])
            print(f"\nExtracted node embeddings for broker {port}:")
            print(f"Shape: {embeddings[port].shape}")
            print(f"Time range: {embeddings[port].index[0]} to {embeddings[port].index[-1]}")
            
        except Exception as e:
            print(f"Error extracting embeddings for broker {port}: {str(e)}")
            continue

    print("\nAll node embeddings extracted successfully!")

    # # 데이터 준비 - 두 번째 시간대 데이터
    # (experiment_config, train_loader, val_loader, test_loader,
    # scaler_X, scaler_y, node_names, feature_names, save_dir) = prepare_experiment_stt_emb(
    #     "merged_1to2", simulation_id, 50003)

    # # 노드 임베딩 불러오기
    # embeddings_df = pd.read_csv(
    #     f'./thesis_sttransformer/unitcase2/embeddings/{simulation_id}/{50003}/node_embeddings.csv',
    #     index_col=0)
    # # 인덱스를 datetime으로 변환
    # embeddings_df.index = pd.to_datetime(embeddings_df.index)

    # # index 형태 확인
    # print("\nEmbeddings DataFrame info:")
    # print(f"Index type: {type(embeddings_df.index[0])}")
    # print(f"First few indices: {embeddings_df.index[:5]}")

    # print("\nDataset indices info:")
    # print(f"Index type: {type(train_loader.dataset.indices[0])}")
    # print(f"First few indices: {train_loader.dataset.indices[:5]}")

    # # 두 데이터의 길이 확인
    # print(f"\nEmbeddings df length: {len(embeddings_df)}")
    # print(f"Train dataset length: {len(train_loader.dataset)}")
    # print(f"Val dataset length: {len(val_loader.dataset)}")
    # print(f"Test dataset length: {len(test_loader.dataset)}")
    
    # # DataLoader 데이터에 임베딩 추가
    # for dataset in [train_loader.dataset, val_loader.dataset, test_loader.dataset]:
    #     # 각 데이터 포인트의 timestamp에 해당하는 임베딩 매칭
    #     dataset.set_embeddings(embeddings_df)



    # 성능 지표와 임베딩 데이터를 사용하여 B3 모델 학습
    # 임베딩이 있는 두 번째 시간대 데이터로 B3 모델 학습
    print("\nTraining B3 model with embeddings...")

    broker_port = 50003  # B3
    save_dir = experiment_paths[broker_port]
    # save_dir = f"./thesis_sttransformer/unitcase2/results/{broker_port}/with_embeddings"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 데이터 준비 - 두 번째 시간대 데이터
    (experiment_config, train_loader, val_loader, test_loader,
        scaler_X, scaler_y, node_names, feature_names, save_dir) = prepare_experiment_stt_emb(
        timestamp, "merged_1to2", simulation_id, broker_port)

    # 노드 임베딩 불러오기
    embeddings_df = pd.read_csv(
        Path(experiment_paths[broker_port]) / 'node_embeddings.csv',
        index_col=0)
    embeddings_df.index = pd.to_datetime(embeddings_df.index)
    d_embedding = embeddings_df.shape[1]  # 임베딩 차원

    # DataLoader 데이터에 임베딩 추가
    for dataset in [train_loader.dataset, val_loader.dataset, test_loader.dataset]:
        dataset.set_embeddings(embeddings_df)

    # experiment_config에 d_embedding 추가
    # experiment_config['d_embedding'] = d_embedding

    # 모델 초기화 및 학습
    experiment = train_model_stt_emb(experiment_config, train_loader, val_loader, test_loader, device, save_dir)

    # 모델 테스트
    print("\n=== Testing B3 model with embeddings ===")
    metrics = test_model(experiment, device, save_dir)
    print(f"Test metrics:", json.dumps(metrics, indent=2))


    # 어텐션 맵 시각화
    print("\n=== Visualizing attention maps ===")

    # # 테스트셋에서 한 배치 데이터 가져오기
    # batch_data, batch_emb, batch_targets = next(iter(test_loader))
    # batch_data = batch_data.to(device)
    # batch_emb = batch_emb.to(device)

    # 특정 timestamp의 데이터 찾기 (선택적)
    specific_time = "2024-11-30 19:00:00"  # 예시 시간
    data_idx = utils.find_data_by_datetime(test_loader.dataset, specific_time)
    if data_idx is not None:
        metrics = test_loader.dataset[data_idx][0].unsqueeze(0).to(device)
        embeddings = test_loader.dataset[data_idx][1].unsqueeze(0).to(device)
        timestamp = test_loader.dataset.indices[data_idx]
    else:
        # 첫 번째 배치의 첫 번째 데이터 사용
        metrics, embeddings, _ = next(iter(test_loader))
        metrics = metrics[0:1].to(device)  # 첫 번째 데이터만 사용
        embeddings = embeddings[0:1].to(device)
        timestamp = test_loader.dataset.indices[0]  # 첫 번째 데이터의 timestamp

    # 전체 레이어에 대한 어텐션 맵 시각화
    num_layers = len(experiment.model.encoder_layers)
    for i in range(num_layers):
        experiment.visualize_attention(
            sample_data=metrics,  # 성능 지표 데이터
            precomputed_emb=embeddings,  # 임베딩 데이터 추가
            node_names=node_names,
            feature_names=feature_names,
            layer_idx=i,
            test_dataset=test_loader.dataset,
            timestamp=timestamp
        )

    print("\nExperiment completed successfully!")
    print(f"Results saved in: {save_dir}")

    
    # 메모리 정리
    del experiment
    clear_memory()
