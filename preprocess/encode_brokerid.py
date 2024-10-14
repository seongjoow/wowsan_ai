import pandas as pd

def encode_brokerid(df) -> pd.DataFrame:
    """
    브로커 ID를 인코딩하는 함수
    "BrokerId_0", "BrokerId_1", "BrokerId_2", ..., "BrokerId_n" 컬럼에 대해서
    각각 "localhost:50002" 형태의 문자열을 50002-50000 계산하여 숫자로 인코딩
    """
    
    # 브로커 ID 컬럼들만 선택
    brokerid_cols = [col for col in df.columns if col.startswith('BrokerId_')]

    # 브로커 ID 컬럼들에 대해 인코딩
    for col in brokerid_cols:
        df[col] = df[col].apply(lambda x: int(x.split(':')[-1]) - 50000 if isinstance(x, str) and ':' in x else x)

    return df