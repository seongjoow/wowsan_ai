# def fill_nan (df):
#     # NaN 값이 존재하는 컬럼 리스트 반환
#     def getNanColumns (df):
#         return df.columns[df.isnull().any()].tolist()
    
#     columns_with_nan = getNanColumns(df)

#     df['HopCount'] = df['HopCount'].fillna(0).astype('int32')

#     # for col in columns_with_nan:
#     #     df[col] = df[col].fillna(0).astype('int32')

