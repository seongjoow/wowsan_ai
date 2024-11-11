import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def handle_grouped_duplicate_time(df, method='mean'):
    """
    같은 날짜(초 단위까지)에 대해 Bn_ 그룹별로 독립적으로 0이 아닌 값들의 평균 또는 최댓값을 계산합니다.
    
    Parameters:
    df (pandas.DataFrame): 처리할 데이터프레임
    method (str): 'mean' 또는 'max' (기본값: 'mean')
    
    Returns:
    pandas.DataFrame: 중복 날짜가 처리된 데이터프레임
    """
    
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
            agg_results = aggregate_group(group, bn, method)
            row_dict.update(agg_results)
        
        # 결과를 데이터프레임에 추가
        result_df = pd.concat([result_df, pd.DataFrame([row_dict])], ignore_index=True)
    
    # time 컬럼을 첫 번째 컬럼으로 이동
    cols = result_df.columns.tolist()
    cols = ['time'] + [col for col in cols if col != 'time']
    result_df = result_df[cols]
    
    return result_df

# def handle_duplicate_dates(df, method='mean'):
#     """
#     같은 날짜(초 단위까지)에 대해 0을 제외한 값들의 평균 또는 최댓값을 계산합니다.
    
#     Parameters:
#     df (pandas.DataFrame): 처리할 데이터프레임
#     method (str): 'mean' 또는 'max' (기본값: 'mean')
    
#     Returns:
#     pandas.DataFrame: 중복 날짜가 처리된 데이터프레임
#     """
    
#     def custom_aggregation(x):
#         # 0을 제외한 값들만 선택
#         non_zero = x[x != 0]
#         if len(non_zero) == 0:
#             return 0  # 모든 값이 0인 경우
        
#         if method == 'mean':
#             return non_zero.mean()
#         elif method == 'max':
#             return non_zero.max()
#         else:
#             raise ValueError("method는 'mean' 또는 'max'여야 합니다.")
    
#     # 숫자형 컬럼만 선택
#     numeric_columns = df.select_dtypes(include=[np.number]).columns
#     numeric_columns = numeric_columns[numeric_columns != 'date']  # date 컬럼 제외
    
#     # 각 컬럼별로 집계 함수 지정
#     agg_dict = {col: custom_aggregation for col in numeric_columns}
    
#     # 날짜별로 그룹화하고 집계
#     result = df.groupby('date', as_index=False).agg(agg_dict)
    
#     return result
