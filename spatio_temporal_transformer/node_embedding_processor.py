import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from queue import Queue
from threading import Thread, Lock
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeNodeEmbeddingProcessor:
    def __init__(self, 
                 model: torch.nn.Module,
                 device: torch.device,
                 num_brokers: int,
                 input_timesteps: int,
                 selected_features: List[str] = None,
                 max_queue_size: int = 1000):
        """
        실시간 노드 임베딩 프로세서 초기화
        
        Args:
            model: 학습된 SpatioTemporalTransformer 모델
            device: 연산 장치 (CPU/GPU)
            num_brokers: 브로커 수
            input_timesteps: 입력으로 사용할 과거 시점 수
            selected_features: 사용할 feature 리스트
            max_queue_size: 메시지 큐의 최대 크기
        """
        
        self.model = model
        self.device = device
        self.num_brokers = num_brokers
        self.input_timesteps = input_timesteps
        self.selected_features = selected_features or ["Throughput", "ResponseTime"]
        
        # 원시 로그 저장소
        self.tick_logs = defaultdict(list)  # 키: timestamp
        self.hop_logs = defaultdict(list)   # 키: timestamp
        self.time_window = timedelta(seconds=input_timesteps)
        
        # 처리된 데이터프레임
        self.processed_df = None
        
        # 메시지 큐와 임베딩 큐
        self.message_queue = Queue(maxsize=max_queue_size)
        self.embedding_queue = Queue()
        
        # 락
        self.data_lock = Lock()
        
        # 스케일러
        self.scaler_X = StandardScaler()
        
        # 처리 스레드 제어
        self.is_running = True
        self.processing_thread = Thread(target=self._process_messages_continuously)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def add_message(self, timestamp: datetime, hop_log: Optional[Dict] = None, 
                   tick_log: Optional[Dict] = None) -> bool:
        """
        새 메시지를 큐에 추가
        
        Args:
            hop_log: Hop 로그 데이터
            tick_log: Tick 로그 데이터
            
        Returns:
            bool: 메시지 추가 성공 여부
        """
        try:
            if self.message_queue.full():
                logger.warning("Message queue is full")
                return False
            
            self.message_queue.put((timestamp, hop_log, tick_log), block=False)
            return True
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return False

    def _process_messages_continuously(self):
        """메시지 처리 스레드의 메인 루프"""
        while self.is_running:
            try:
                if self.message_queue.empty():
                    # logger.info("Message queue is empty")
                    continue
                try:
                    timestamp, hop_log, tick_log = self.message_queue.get(block=False)
                except Exception as e:
                    logger.error(f"Error getting message: {e}")
                    continue
                with self.data_lock:
                    # 로그 저장
                    try:
                        if hop_log:
                            if self.hop_logs.get(timestamp) is None:
                                self.hop_logs[timestamp] = [hop_log]
                            else:
                                self.hop_logs[timestamp].append(hop_log)
                        if tick_log:
                            if self.tick_logs.get(timestamp) is None:
                                self.tick_logs[timestamp] = [tick_log]
                            else:
                                self.tick_logs[timestamp].append(tick_log)
                    except Exception as e:
                        logger.error(f"Error saving log: {e}")
                    
                    # 오래된 데이터 제거
                    cutoff_time = timestamp - self.time_window
                    self._cleanup_old_data(cutoff_time)
                    
                    # 시계열 데이터 생성 및 전처리
                    df = self._create_time_series_data(timestamp)
                    logger.info(f"시계열 데이터 생성 및 전처리")
                    if df is not None:
                        # 모델 입력 준비
                        model_input = self._prepare_data_for_model(df)
                        if model_input is not None:
                            # 노드 임베딩 계산
                            embeddings = self.get_node_embedding(model_input)
                            
                            # 노드 임베딩 값 출력
                            logger.info(f"Node embeddings: {embeddings}")                            

                            logger.info(f"노드 임베딩 계산")
                            if embeddings is not None:
                                self.embedding_queue.put(embeddings)
                try:
                    self.message_queue.task_done()
                except Exception as e:
                    logger.error(f"Error marking message as done: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue

    def _cleanup_old_data(self, cutoff_time: datetime):
        """오래된 로그 제거"""
        try:
            self.tick_logs = {t: logs for t, logs in self.tick_logs.items() 
                             if t >= cutoff_time}
            self.hop_logs = {t: logs for t, logs in self.hop_logs.items() 
                            if t >= cutoff_time}
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")


    def _create_time_series_data(self, current_time: datetime) -> Optional[pd.DataFrame]:
        """
        시간 윈도우 내의 데이터 처리
        누락된 tick log 체크 및 처리 추가
        """
        try:
            start_time = current_time - self.time_window
            
            # 1. 시간 범위 내 모든 로그 수집
            tick_times = set(self.tick_logs.keys())
            hop_times = set(self.hop_logs.keys())
            
            # 2-1. 예상되는 모든 초 생성 (tick log가 있어야 할 시간들)
            expected_times = set()
            current_time = pd.Timestamp(current_time)
            time_pointer = pd.Timestamp(start_time)
            while time_pointer <= current_time:
                expected_times.add(time_pointer)
                time_pointer += pd.Timedelta(seconds=1)
            
            # 2-2. 누락된 tick log 시간 확인
            missing_times = expected_times - tick_times
            if missing_times:
                logger.warning(f"Missing tick logs at: {missing_times} times count: {len(missing_times)}")
                
                # tick log가 2초 이상 연속으로 누락되었는지 확인
                consecutive_missing = 0
                for t in sorted(expected_times):
                    if t in missing_times:
                        consecutive_missing += 1
                        if consecutive_missing >= 2:
                            logger.error("Too many consecutive tick logs missing")
                            return None
                    else:
                        consecutive_missing = 0

            # # 3. Hop 로그 처리
            # hop_data = []
            # for timestamp in sorted(hop_times):
            #     if start_time <= timestamp <= current_time:
            #         if timestamp in self.hop_logs:
            #             hop_df = self._preprocess_hop_log(pd.DataFrame(self.hop_logs[timestamp]))
            #             hop_data.append(hop_df)
            # logger.info(f"3. Hop 로그 처리")

            hop_data = []
            for timestamp in sorted(hop_times):
                if start_time <= timestamp <= current_time:
                    if timestamp in self.hop_logs:
                        hop_data.append(pd.DataFrame(self.hop_logs[timestamp]))

            full_hop_df = pd.concat(hop_data) if hop_data else None

            if full_hop_df is not None:
                full_hop_df = self._preprocess_hop_log(full_hop_df)
                logger.info(f"3-1. Hop 로그 처리")
                full_hop_df = self._handle_duplicate_times(full_hop_df)
                logger.info(f"3-2. 중복 처리")
            
            # # 4. Hop 데이터 합치고 중복 처리
            # full_hop_df = pd.concat(hop_data) if hop_data else None
            # if full_hop_df is not None:
            #     full_hop_df = self._handle_duplicate_times(full_hop_df)  # 중복 처리
            #     # 중복 검사
            #     duplicate_count = full_hop_df['time'].duplicated().sum()
            #     if duplicate_count > 0:
            #         logger.warning(f"Found {duplicate_count} duplicates in hop_df after handling")
            # logger.info(f"4. Hop 데이터 합치고 중복 처리")

            # 5. Tick 로그 처리
            tick_data = []
            for timestamp in sorted(tick_times):
                if start_time <= timestamp <= current_time:
                    if timestamp in self.tick_logs:
                        tick_df = pd.DataFrame(self.tick_logs[timestamp])
                        tick_data.append(tick_df)
            
            full_tick_df = pd.concat(tick_data) if tick_data else None
            logger.info(f"5. Tick 로그 처리")
            
            # 6. Merge
            if full_hop_df is not None or full_tick_df is not None:
                merged_df = self._merge_logs(full_hop_df, full_tick_df)
                
                # 7. 누락된 시간에 대한 보간
                if missing_times:
                    merged_df.set_index('time', inplace=True)
                    merged_df.index = pd.to_datetime(merged_df.index)  # datetime으로 변환
                    merged_df = merged_df.sort_index()
                    merged_df = merged_df.interpolate(method='time')
                    merged_df.reset_index(inplace=True)
                
                logger.info(f"6. 로그 병합")
                return merged_df

            return None
            
            # # 4. 데이터 처리 (기존 로직)
            # logger.info(f"4. 데이터 처리 (기존 로직)")
            # processed_data = []
            # for timestamp in sorted(tick_times | hop_times):
            #     if start_time <= timestamp <= current_time:
            #         hop_df = None
            #         if timestamp in self.hop_logs:
            #             hop_df = self._preprocess_hop_log(
            #                 pd.DataFrame(self.hop_logs[timestamp])
            #             )
            #         logger.info(f"4-1. preprocessed hop log")
            #         tick_df = None
            #         if timestamp in self.tick_logs:
            #             tick_df = pd.DataFrame(self.tick_logs[timestamp])
            #         logger.info(f"4-2. preprocessed tick log")
                    
            #         if hop_df is not None or tick_df is not None:
            #             merged_df = self._merge_logs(hop_df, tick_df)
            #             processed_data.append(merged_df)
            #         logger.info(f"4-3. merged log")
            
            # if not processed_data:
            #     return None
            
            # # 5. 데이터 병합 및 중복 처리 (기존 로직)
            # logger.info(f"5. 데이터 병합 및 중복 처리 (기존 로직)")
            # df = pd.concat(processed_data, ignore_index=True)
            # df = self._handle_duplicate_times(df)
            
            # # 6. 누락된 시간에 대한 처리
            # logger.info(f"6. 누락된 시간에 대한 처리")
            # if missing_times:
            #     df.set_index('time', inplace=True)
            #     df = df.sort_index()
                
            #     # 앞뒤 값의 평균으로 보간
            #     # 주의: 이 부분은 실제 사용 케이스에 따라 다른 전략 선택 가능
            #     # - 앞의 값 사용
            #     # - 뒤의 값 사용
            #     # - 평균값 사용
            #     # - 기계학습 모델로 예측
            #     df = df.interpolate(method='time')
                
            #     # 인덱스 리셋
            #     df.reset_index(inplace=True)
            
            # return df
            
        except Exception as e:
            logger.error(f"Error creating time series data: {e}")
            return None

    def _preprocess_hop_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """기존 hop_log 전처리 코드"""
        """Hop 로그 전처리"""
        # 브로커별 세부 정보 컬럼 초기화
        for i in range(1, self.num_brokers + 1):
            df[f'B{i}_Cpu'] = np.nan
            df[f'B{i}_Memory'] = np.nan
            df[f'B{i}_QueueLength'] = np.nan
            df[f'B{i}_QueueTime'] = np.nan
            df[f'B{i}_ServiceTime'] = np.nan
            df[f'B{i}_ResponseTime'] = np.nan
            df[f'B{i}_InterArrivalTime'] = np.nan
            df[f'B{i}_Throughput'] = np.nan
            df[f'B{i}_m'] = np.nan
            # df[f'B{i}_TimestampDiff'] = np.nan

        # DataFrame의 각 행 처리
        for idx, row in df.iterrows():
            if isinstance(row['PerformanceInfo'], list):
                for broker_info in row['PerformanceInfo']:
                    broker_id = broker_info.get('BrokerId', '')
                    if broker_id:
                        port_number = int(broker_id.split(':')[-1])
                        broker_index = port_number - 50000
                        
                        # 성능 메트릭 저장
                        df.at[idx, f'B{broker_index}_Cpu'] = float(broker_info.get('Cpu', np.nan))
                        df.at[idx, f'B{broker_index}_Memory'] = float(broker_info.get('Memory', np.nan))
                        df.at[idx, f'B{broker_index}_QueueLength'] = float(broker_info.get('QueueLength', np.nan))
                        df.at[idx, f'B{broker_index}_QueueTime'] = float(broker_info.get('QueueTime', np.nan))
                        df.at[idx, f'B{broker_index}_ServiceTime'] = float(broker_info.get('ServiceTime', np.nan))
                        df.at[idx, f'B{broker_index}_ResponseTime'] = float(broker_info.get('ResponseTime', np.nan))
                        df.at[idx, f'B{broker_index}_InterArrivalTime'] = float(broker_info.get('InterArrivalTime', np.nan))
                        df.at[idx, f'B{broker_index}_Throughput'] = float(broker_info.get('Throughput', np.nan))
                        
                        # # m 값 추출
                        # timestamp = broker_info.get('Timestamp', '')
                        # if timestamp and 'm=' in timestamp:
                        #     m_value = float(timestamp.split('m=')[-1])
                        #     df.at[idx, f'B{broker_index}_m'] = m_value

        # PerformanceInfo 컬럼 삭제
        df.drop('PerformanceInfo', axis=1, inplace=True)
        return df

    def _merge_logs(self, hop_df: pd.DataFrame, tick_df: pd.DataFrame) -> pd.DataFrame:
        """기존 로그 병합 코드"""
        """Tick 로그와 Hop 로그 병합"""
        broker_index = 3  # 실제 상황에 맞게 조정 필요
        
        # Tick 로그 컬럼 매핑
        tick_columns_mapping = {
            'Cpu': f'B{broker_index}_Cpu',
            'Memory': f'B{broker_index}_Memory',
            'QueueLength': f'B{broker_index}_QueueLength',
            'QueueTime': f'B{broker_index}_QueueTime',
            'ServiceTime': f'B{broker_index}_ServiceTime',
            'ResponseTime': f'B{broker_index}_ResponseTime',
            'InterArrivalTime': f'B{broker_index}_InterArrivalTime',
            'Throughput': f'B{broker_index}_Throughput',
        }
        
        tick_df = tick_df.rename(columns=tick_columns_mapping)
        # return tick_df.merge(hop_df, on='time', how='left', suffixes=('', '_hop'))

        # 두 DataFrame 출력
        logger.info(f"Tick DataFrame: {tick_df}")
        logger.info(f"Hop DataFrame: {hop_df}")
        
        
        

        # Merge the two DataFrames
        result_df = tick_df.merge(hop_df, on='time', how='left', suffixes=('', '_hop'))

        # Extract column names
        tick_columns = tick_df.columns.tolist()
        hop_columns = hop_df.columns.tolist()

        # Columns to be replaced in tick_df
        replace_columns = [col for col in tick_columns if col != 'time']

        # Replace tick_df values with hop_df values if they exist in hop_df
        for col in replace_columns:
            hop_col = col + '_hop'
            if hop_col in result_df.columns:
                result_df[col] = result_df[hop_col].combine_first(result_df[col])
                result_df.drop(columns=[hop_col], inplace=True)

        # Additional columns from hop_df
        additional_columns = [col for col in hop_columns if col not in tick_columns]

        # Fill the remaining columns from hop_df
        result_df.update(hop_df.set_index('time')[additional_columns])

        return result_df

    def _handle_duplicate_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """기존 중복 처리 코드"""
        """중복된 시간 처리"""
        def get_group_columns(df, prefix):
            """특정 Bn_ 프리픽스를 가진 컬럼들을 반환"""
            return [col for col in df.columns if col.startswith(prefix)]
        
        def has_nonzero_values(row, columns):
            """주어진 컬럼들 중에서 0이 아닌 값이 있는지 확인"""
            return any(row[columns] != 0)
        
        def aggregate_group(group, bn_prefix, method):
            """각 Bn_ 그룹에 대해 집계 수행"""
            columns = get_group_columns(group, bn_prefix)
            
            # 해당 Bn_ 그룹의 컬럼들 중 하나라도 0이 아닌 값이 있는 행만 선택
            valid_rows = group[group.apply(lambda row: has_nonzero_values(row, columns), axis=1)]
            
            if len(valid_rows) == 0:
                # 모든 행이 해당 Bn_ 그룹에서 0인 경우
                return pd.Series({col: 0 for col in columns})
            
            # 선택된 행들에 대해 평균 또는 최댓값 계산
            if method == 'mean':
                return valid_rows[columns].mean()
            else:  # method == 'max'
                return valid_rows[columns].max()
        
        # 결과를 저장할 빈 데이터프레임 생성
        result_df = pd.DataFrame()
        
        # time로 그룹화
        grouped = df.groupby('time')
        
        # 각 time 그룹에 대해 처리
        for time, group in grouped:
            row_dict = {'time': time}
            
            # 각 Bn_ 그룹별로 처리
            for bn in ['B1_', 'B2_', 'B3_', 'B4_', 'B5_']:
                agg_results = aggregate_group(group, bn, 'mean')
                row_dict.update(agg_results)
            
            # 결과를 데이터프레임에 추가
            result_df = pd.concat([result_df, pd.DataFrame([row_dict])], ignore_index=True)
        
        # time 컬럼을 첫 번째 컬럼으로 이동
        cols = result_df.columns.tolist()
        cols = ['time'] + [col for col in cols if col != 'time']
        result_df = result_df[cols]
        
        return result_df

    def _prepare_data_for_model(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """기존 모델 입력 준비 코드"""
        """DataFrame을 모델 입력 형태로 변환"""
        try:
            #  # 시간 출력
            # logger.info("Unique timestamps in data:")
            # timestamps = df['time'].unique()
            # for ts in sorted(timestamps):
            #     logger.info(f"Timestamp: {ts}")
            # logger.info(f"Total unique timestamps: {len(timestamps)}")

            # # time별 데이터 개수 출력
            # logger.info("Records per timestamp:")
            # counts = df['time'].value_counts().sort_index()
            # for time, count in counts.items():
            #     logger.info(f"Timestamp {time}: {count} records")
                
            # time 컬럼 제외하고 feature만 선택
            # feature_cols = [col for col in df.columns if col != 'time']
            feature_cols = []
            for feature in self.selected_features:
                feature_cols.extend([col for col in df.columns if col.endswith(f'_{feature}')])
            df_features = df[feature_cols]
            
            # 스케일링 적용
            scaled_data = self.scaler_X.fit_transform(df_features)

            # # 데이터 reshape: [timesteps, total_features] -> [timesteps, num_nodes, num_features]
            # num_timesteps = min(len(scaled_data), self.input_timesteps)
            
            # 마지막 timesteps 개수만큼만 선택
            if len(scaled_data) > self.input_timesteps:
                scaled_data = scaled_data[-self.input_timesteps:]

            # # reshape 전에 크기 확인
            # expected_size = self.input_timesteps * self.num_brokers * len(self.selected_features)
            # current_size = scaled_data.size
            
            # logger.info(f"Expected size: {expected_size}, Current size: {current_size}")
            # logger.info(f"Shape before reshape: {scaled_data.shape}")

            # reshape
            x = scaled_data.reshape(self.input_timesteps, self.num_brokers, len(self.selected_features))
            
            # 텐서로 변환 (배치 차원은 추가하지 않음) 및 모델 입력 형태로 변환
            x = torch.FloatTensor(x).to(self.device)
            
            logger.info(f"Final shape: {x.shape}")
            return x
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None

    def get_node_embedding(self, model_input: torch.Tensor) -> Optional[np.ndarray]:
        """기존 노드 임베딩 계산 코드"""
        """노드 임베딩 계산"""
        try:
            with torch.no_grad():
                embeddings = self.model.get_node_embedding(model_input.unsqueeze(0))
            return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error calculating node embedding: {e}")
            return None

    def get_latest_embedding(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        최신 임베딩 가져오기
        
        Args:
            timeout: 대기 시간 (초)
            
        Returns:
            Optional[np.ndarray]: 최신 노드 임베딩 또는 None
        """
        try:
            return self.embedding_queue.get(timeout=timeout)
        except Queue.Empty:
            return None

    def stop(self):
        """프로세서 정상 종료"""
        self.is_running = False
        self.processing_thread.join()
        logger.info("Node embedding processor stopped")


# 주요 변경사항:
# 1. 시간 기반 로그 저장소 구현 (tick_logs, hop_logs)
# 2. 지정된 시간 윈도우의 모든 로그 처리
# 3. 1초 단위 리샘플링 적용
# 4. 결측치 처리 로직 추가

# 사용 예시:
# 초기화
# processor = TimeSeriesNodeEmbeddingProcessor(
#     model=model,
#     device=device,
#     num_brokers=5,
#     input_timesteps=10
# )

# # 메시지 처리
# timestamp = datetime.now()
# processor.add_message(timestamp, hop_log=hop_log)
# processor.add_message(timestamp, tick_log=tick_log)

# # 임베딩 확인
# embeddings = processor.get_latest_embedding()
