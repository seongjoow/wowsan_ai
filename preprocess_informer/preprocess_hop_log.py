import json
import numpy as np
import pandas as pd
from dateutil import parser

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# # json 파일을 열고 각 줄을 개별적으로 파싱
# data = []
# with open('hopLogger.json', 'r') as file:
#     for line in file:
#         data.append(json.loads(line))

# # 데이터를 DataFrame으로 변환
# df = pd.DataFrame(data)
# type(df)


def parse_performance_info(df):
    """
    메시지의 전달 경로 상에 있는 브로커에서 수집된 로그를 최근순으로 파싱.
    
    """

    max_hops = 0
    rows = []
    keys = set()

    # PerformanceInfo 컬럼의 리스트 길이를 HopCount라는 새로운 컬럼으로 추가 (자료형: int)
    df['HopCount'] = df['PerformanceInfo'].apply(lambda x: int(len(x)))
    df['HopCount'] = df['HopCount'].astype(int)

    for tran in df['PerformanceInfo']:
        # tran이 문자열인지 확인하고, 맞다면 json.loads를 사용
        parsed = json.loads(tran) if isinstance(tran, str) else tran
        for hop in parsed:
            keys.update(hop.keys())

    for tran in df['PerformanceInfo']:
        # tran이 문자열인지 확인하고, 맞다면 json.loads를 사용
        parsed = json.loads(tran) if isinstance(tran, str) else tran
        row = {}
        for idx, hop in enumerate(reversed(parsed)): # 리스트의 마지막 요소부터 시작
            for key, value in hop.items():
                row[f'{key}_{idx}'] = value
        rows.append(row)
        max_hops = max(max_hops, len(parsed))

    # 패딩: 모든 행을 가장 많은 홉 수에 맞추어 확장
    final_rows = []
    for row in rows:
        padded_row = {f'{key}_{i}': row.get(f'{key}_{i}', np.nan) for i in range(max_hops) for key in keys}
        final_rows.append(padded_row)

    # 최종 DataFrame 생성
    performance_df = pd.DataFrame(final_rows)

    # 원래 DataFrame과 합치기
    result_df = pd.concat([df[['time', 'Node', 'level', 'msg', 'HopCount']], performance_df], axis=1)

    return result_df


def process_timestamp(hop_df):
    """
    Process the Timestamp columns in the dataframe to calculate the difference 
    in 'm' values and set the 'Timestamp_0' column to 0.
    
    Parameters:
    hop_df (pd.DataFrame): DataFrame containing Timestamp columns along with other data.
    
    Returns:
    pd.DataFrame: Processed DataFrame with calculated 'm' value differences.
    """
    
    def extract_m_value(Timestamp_n):
        if pd.isna(Timestamp_n):
            return np.nan
        if 'm=' in Timestamp_n:
            return float(Timestamp_n.split('m=')[1])
        else:
            return np.nan

    # Timestamp_n 형식의 컬럼들만 선택
    timestamp_cols = [col for col in hop_df.columns if col.startswith('Timestamp_')]

    # Timestamp 컬럼들만 선택하여 새로운 데이터프레임 생성
    m_values_df = hop_df[timestamp_cols].applymap(extract_m_value)

    # 각 행을 개별적으로 처리하여 각 컬럼의 값을 다음 컬럼의 m 값과의 차이로 변환
    def compute_diff(row):
        for i in range(0, len(row)-1):
            if pd.notna(row[i+1]):
                next_value = row[i+1]
                row[i] = row[i] - next_value
            else :
                row[i] = 0
                break
        row[len(row)-1] = 0
        return row

    # 데이터프레임에 대해 위의 함수를 적용
    m_values_df = m_values_df.apply(compute_diff, axis=1)

    # # Timestamp_0 컬럼의 값을 0으로 설정
    # m_values_df['Timestamp_0'] = 0.0

    # 원래 데이터프레임의 나머지 컬럼들과 병합
    result_df = hop_df.drop(columns=timestamp_cols).join(m_values_df)

    return result_df
