import json
import numpy as np
import pandas as pd
from dateutil import parser

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

def parse_performance_info(num_brokers : int, df : pd.DataFrame) -> pd.DataFrame:
    # 브로커 별 세부 정보 컬럼을 초기화
    for i in range(1, num_brokers + 1):
        df[f'B{i}_Cpu'] = np.nan
        df[f'B{i}_Memory'] = np.nan
        df[f'B{i}_QueueLength'] = np.nan
        df[f'B{i}_QueueTime'] = np.nan
        df[f'B{i}_ServiceTime'] = np.nan
        df[f'B{i}_ResponseTime'] = np.nan
        df[f'B{i}_InterArrivalTime'] = np.nan
        df[f'B{i}_Throughput'] = np.nan
        df[f'B{i}_m'] = np.nan
        df[f'B{i}_TimestampDiff'] = np.nan

    # 'm' 값을 추출하는 함수
    def extract_m_value(timestamp: str) -> float:
        try:
            # 'm=' 이후의 값을 추출
            m_value = timestamp.split("m=")[-1]
            return float(m_value)
        except (IndexError, ValueError):
            return np.nan

    # DataFrame의 각 행을 반복 처리
    for idx, row in df.iterrows():
        if isinstance(row['PerformanceInfo'], list):
            for broker_info in row['PerformanceInfo']:
                broker_id = broker_info.get('BrokerId', '')
                if broker_id:
                    # broker_id에서 포트 번호를 추출하고 50000을 뺌
                    port_number = int(broker_id.split(':')[-1])
                    broker_index = port_number - 50000
                    # 해당 브로커 인덱스의 각 세부 정보 컬럼에 데이터 저장
                    df.at[idx, f'B{broker_index}_Cpu'] = broker_info.get('Cpu', np.nan)
                    df.at[idx, f'B{broker_index}_Memory'] = broker_info.get('Memory', np.nan)
                    df.at[idx, f'B{broker_index}_QueueLength'] = broker_info.get('QueueLength', np.nan)
                    df.at[idx, f'B{broker_index}_QueueTime'] = broker_info.get('QueueTime', np.nan)
                    df.at[idx, f'B{broker_index}_ServiceTime'] = broker_info.get('ServiceTime', np.nan)
                    df.at[idx, f'B{broker_index}_ResponseTime'] = broker_info.get('ResponseTime', np.nan)
                    df.at[idx, f'B{broker_index}_InterArrivalTime'] = broker_info.get('InterArrivalTime', np.nan)
                    df.at[idx, f'B{broker_index}_Throughput'] = broker_info.get('Throughput', np.nan)
                    # df.at[idx, f'B{broker_index}_Timestamp'] = broker_info.get('Timestamp', np.nan)                

                    # 현재 요소의 m 값 추출
                    current_m_value = extract_m_value(broker_info.get('Timestamp', ''))
                    df.at[idx, f'B{broker_index}_m'] = current_m_value
                    
                    # 다음 요소가 있으면 m 값 차이 계산, 없으면 0
                    if i + 1 < len(row['PerformanceInfo']):
                        next_m_value = extract_m_value(row['PerformanceInfo'][i + 1].get('Timestamp', ''))
                        timestamp_diff = next_m_value - current_m_value
                    else:
                        timestamp_diff = np.nan

                    # Timestamp 차이값 저장
                    df.at[idx, f'B{broker_index}_TimestampDiff'] = timestamp_diff

        else:
            print(f"Unexpected data format at index {idx}: {row['PerformanceInfo']}")

    # 'PerformanceInfo' 컬럼 삭제
    df.drop('PerformanceInfo', axis=1, inplace=True)


    return df


# def process_timestamp()