import pandas as pd
import numpy as np
import logging
from typing import Union, Tuple
from datetime import datetime

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
