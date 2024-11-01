import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import os

class CustomTimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, target_window, target_node, target_feature):
        self.data = data
        self.input_window = input_window
        self.target_window = target_window
        self.target_node = target_node
        self.target_feature = target_feature
        
        # # 입력 데이터에서 제외할 컬럼 생성
        # self.target_columns = [f"{target_node}_{f}" for f in target_features]

    def __len__(self):
        return len(self.data) - self.input_window - self.target_window + 1

    def __getitem__(self, index):
        # # 입력 데이터에서 타겟 특성 제외
        # input_data = self.data.iloc[index:index+self.input_window].drop(columns=self.target_columns).values.flatten()
        
        input_data = self.data[index:index+self.input_window].values.flatten()
        # target_cols = [f"{self.target_node}_{f}" for f in self.target_features]
        target_col = f"{self.target_node}_{self.target_feature}"
        target_data = self.data[index+self.input_window:index+self.input_window+self.target_window][target_col].values

        return torch.FloatTensor(input_data), torch.FloatTensor(target_data)
    
    def save_to_csv(self, file_path):
        all_input_data = []
        all_target_data = []

        for i in range(len(self)):
            input_data, target_data = self[i]
            all_input_data.append(input_data.numpy())
            all_target_data.append(target_data.numpy())

        # 입력 데이터의 컬럼 이름 생성
        input_columns = []
        for t in range(self.input_window):
            for col in self.data.columns:
                input_columns.append(f"t{t}{col}")

        target_columns = [f"t{t}{self.target_node}_{self.target_feature}" for t in range(self.input_window, self.input_window + self.target_window)]

        df = pd.DataFrame(all_input_data, columns=input_columns)
        df = pd.concat([df, pd.DataFrame(all_target_data, columns=target_columns)], axis=1)

        df.to_csv(file_path, index=False)
        print(f"Dataset saved to {file_path}")

def load_and_prepare_data(file_path, input_window, target_window, target_node, target_feature, test_size=0.2):
    # # CSV 파일 로드
    # df = pd.read_csv(file_path, index_col=0)
    
    # # # 데이터 정규화
    # # scaler = StandardScaler()
    # # df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    # # 데이터셋 생성
    # # dataset = CustomTimeSeriesDataset(df_scaled, input_window, target_window, target_node, target_features)
    # dataset = CustomTimeSeriesDataset(df, input_window, target_window, target_node, target_features)
    
    # # 학습 및 테스트 세트 분할
    # train_size = int((1 - test_size) * len(dataset))
    # test_size = len(dataset) - train_size
    # # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # train_df, test_df = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # # 데이터 정규화
    # scaler = StandardScaler()
    # train_dataset = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    # test_dataset = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_df.index)

    # return train_dataset, test_dataset, scaler

    # CSV 파일 로드
    df = pd.read_csv(file_path, index_col=0)
    path_mergeddf = './preprocessed_data_slotted/89/'

    # drop_columns = [ 
    #     'date',
    # #     'BrokerId_0',
    # #     'BrokerId_1',
    # #     'BrokerId_2'
    # ]
    # # 불필요한 컬럼 제거
    # df.drop(columns=drop_columns, inplace=True)

    # input, target feature로 사용할 column 선택
    selected_features = ["Cpu", "Memory", "Throughput", "ResponseTime"]
    df = df[[col for col in df.columns if any(col.endswith('_' + feature) for feature in selected_features)]]
    # # 결과 출력
    # print(df)
    # csv 파일로 저장
    df.to_csv(path_mergeddf+'selected_merged_df.csv', date_format='%Y-%m-%d %H:%M:%S', index=False)

    # 학습 및 테스트 세트 분할
    train_size = int((1 - test_size) * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # 데이터 정규화
    scaler = StandardScaler()
    train_df_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    test_df_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_df.index)

    # 데이터셋 생성
    # dataset = CustomTimeSeriesDataset(df_scaled, input_window, target_window, target_node, target_features)
    train_dataset = CustomTimeSeriesDataset(train_df_scaled, input_window, target_window, target_node, target_feature)
    test_dataset = CustomTimeSeriesDataset(test_df_scaled, input_window, target_window, target_node, target_feature)

    return train_dataset, test_dataset, scaler

def create_dataloaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def save_datasets(train_dataset, test_dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.save_to_csv(os.path.join(output_dir, "train_dataset.csv"))
    test_dataset.save_to_csv(os.path.join(output_dir, "test_dataset.csv"))

# 사용 예시
if __name__ == "__main__":
    file_path = "./preprocessed_data_slotted/89/merged_df.csv"
    input_window = 3  # t1, t2, t3를 입력으로 사용
    target_window = 5  # t4부터 t9까지 예측
    target_node = "B2"  # B2 노드 예측
    # target_features = ["Throughput", "ResponseTime"]  # f1과 f2 성능 지표 예측
    target_feature = "Throughput"
    batch_size = 32
    output_dir = "./dataset_slotted"

    train_dataset, test_dataset, scaler = load_and_prepare_data(
        file_path, input_window, target_window, target_node, target_feature
    )

    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size)

    # 데이터 형태 확인
    for batch_x, batch_y in train_loader:
        print("Input shape:", batch_x.shape)
        print("Target shape:", batch_y.shape)
        break

    print("Number of training samples:", len(train_dataset))
    print("Number of test samples:", len(test_dataset))

    # 데이터셋을 CSV 파일로 저장
    save_datasets(train_dataset, test_dataset, output_dir)