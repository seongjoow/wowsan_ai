import pandas as pd

def merge_dataframes(hop_df, tick_df) -> pd.DataFrame:

    # tick_df의 컬럼 이름 변경 
    tick_df = tick_df.rename(columns={
        'Cpu': 'Cpu_0',
        'InterArrivalTime': 'InterArrivalTime_0',
        'Memory': 'Memory_0',
        'QueueLength': 'QueueLength_0',
        'QueueTime': 'QueueTime_0',
        'ResponseTime': 'ResponseTime_0',
        'ServiceTime': 'ServiceTime_0',
        'Throughput': 'Throughput_0'
    })
    
    # Merge the two DataFrames
    result_df = tick_df.merge(hop_df, on='time', how='left', suffixes=('', '_hop'))

    # Extract column names
    tick_columns = tick_df.columns.tolist()
    hop_columns = hop_df.columns.tolist()

    # Columns to be replaced in tick_df
    replace_columns = [col for col in tick_columns if col != 'time']

    # # Replace tick_df values with hop_df values if they exist in hop_df
    # for col in replace_columns:
    #     result_df[col] = result_df[col + '_hop'].combine_first(result_df[col])
    #     result_df.drop(columns=[col + '_hop'], inplace=True)

    # Replace tick_df values with hop_df values if they exist in hop_df
    for col in replace_columns:
        hop_col = col + '_hop'
        if hop_col in result_df.columns:
            result_df[col] = result_df[hop_col].combine_first(result_df[col])
            result_df.drop(columns=[hop_col], inplace=True)

    # Additional columns from hop_df
    additional_columns = [col for col in hop_columns if col not in tick_columns]

    # # Fill the remaining columns from hop_df
    # result_df.update(hop_df.set_index('time')[additional_columns])

    return result_df